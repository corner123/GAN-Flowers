"""VAE-GAN for Text-to-Image Generation (v2 — CLIP + Cross-Attention).

Architecture:
  TextEncoder:  CLIP ViT-B/32 → condition vector (512d)
  Encoder:      128x128x3 → μ, logσ² (latent_dim=256)
  Decoder:      z(256) + condition(512) → 128×128×3   (generator)
  Discriminator: 128×128×3 + condition(512) → real/fake logit

v2 improvements:
  - CLIP text encoder (512d contextual embeddings, not GloVe 50d)
  - Cross-attention layers in decoder for fine-grained text-image alignment
  - Matching loss (text-image cosine similarity) to enforce conditioning
  - Stronger adversarial loss weight
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ═══════════════════════════════════════════════
#  Building blocks
# ═══════════════════════════════════════════════

class ResBlockDown(nn.Module):
    """Residual block with stride-2 downsampling."""
    def __init__(self, in_ch, out_ch, norm=True, leaky=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 2, 0, bias=False)

        if leaky:
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)
        self.norm1 = nn.InstanceNorm2d(out_ch) if norm else nn.Identity()
        self.norm2 = nn.InstanceNorm2d(out_ch) if norm else nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        s = self.skip(x)
        return self.act(h + s)


class ResBlockUp(nn.Module):
    """Residual block with nearest-upsample then conv (no checkerboard)."""
    def __init__(self, in_ch, out_ch, condition_dim, norm=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

        if norm:
            self.norm1 = nn.InstanceNorm2d(out_ch)
            self.norm2 = nn.InstanceNorm2d(out_ch)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.act = nn.ReLU(inplace=True)

        # FiLM: condition → per-channel scale & bias
        self.cond_proj = nn.Linear(condition_dim, out_ch * 2)

    def forward(self, x, condition):
        x_up = self.upsample(x)
        h = self.act(self.norm1(self.conv1(x_up)))
        h = self.norm2(self.conv2(h))

        # Condition modulation
        scale, bias = self.cond_proj(condition).chunk(2, dim=1)
        h = h * (1.0 + scale.view(-1, scale.shape[1], 1, 1))
        h = h + bias.view(-1, bias.shape[1], 1, 1)

        s = self.upsample(self.skip(x))
        return self.act(h + s)


class ResBlockPlain(nn.Module):
    """Residual block without spatial size change."""
    def __init__(self, ch, leaky=True):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.act = nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)
        self.norm1 = nn.InstanceNorm2d(ch)
        self.norm2 = nn.InstanceNorm2d(ch)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + x)


# ═══════════════════════════════════════════════
#  Cross-Attention (for text conditioning)
# ═══════════════════════════════════════════════

class CrossAttention(nn.Module):
    """Cross-attention: image features attend to text embeddings.

    Q from image features, K/V from text embedding.
    Applied on spatial feature maps reshaped to sequence.
    """
    def __init__(self, feat_dim, text_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(text_dim, feat_dim)
        self.v_proj = nn.Linear(text_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, feat, text_embed):
        """
        feat: (B, C, H, W) spatial features
        text_embed: (B, text_dim) or (B, N, text_dim) token embeddings
        """
        B, C, H, W = feat.shape
        # Reshape to (B, H*W, C)
        feat_flat = feat.view(B, C, H * W).permute(0, 2, 1)
        residual = feat_flat

        # If text_embed is 2D, add a token dimension
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)  # (B, 1, text_dim)

        # Q from image, K/V from text
        q = self.q_proj(feat_flat).view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(text_embed).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(text_embed).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, H * W, C)

        out = self.out_proj(out)
        out = self.norm(out + residual)

        # Reshape back to (B, C, H, W)
        return out.permute(0, 2, 1).view(B, C, H, W)


# ═══════════════════════════════════════════════
#  Text Encoder (CLIP)
# ═══════════════════════════════════════════════

class TextEncoder(nn.Module):
    """CLIP ViT-B/32 text encoder → 512d condition vector.

    Frozen CLIP produces high-quality text embeddings that capture
    semantic meaning far better than GloVe mean-pooling.
    """
    def __init__(self, text_dim=512, condition_dim=512):
        super().__init__()
        # Projection from CLIP dim to condition dim
        self.proj = nn.Sequential(
            nn.Linear(text_dim, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(self, text_embed):
        """text_embed: (B, 512) CLIP embedding"""
        return self.proj(text_embed)


# ═══════════════════════════════════════════════
#  Image Encoder
# ═══════════════════════════════════════════════

class Encoder(nn.Module):
    """128×128×3 → μ, logσ² (each latent_dim)."""
    def __init__(self, base_ch=64, latent_dim=256):
        super().__init__()
        c = base_ch
        self.head = nn.Conv2d(3, c, 3, 1, 1, bias=False)
        # 128 → 64 → 32 → 16 → 8 → 4
        self.d1 = ResBlockDown(c, c * 2)     # 128 → 64
        self.d2 = ResBlockDown(c * 2, c * 4) # 64  → 32
        self.d3 = ResBlockDown(c * 4, c * 8) # 32  → 16
        self.d4 = ResBlockDown(c * 8, c * 8) # 16  → 8
        self.d5 = ResBlockDown(c * 8, c * 8) # 8   → 4
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(c * 8 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(c * 8 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.head(x)
        h = self.d1(h)
        h = self.d2(h)
        h = self.d3(h)
        h = self.d4(h)
        h = self.d5(h)
        h = self.flatten(h)
        return self.fc_mu(h), self.fc_logvar(h)


# ═══════════════════════════════════════════════
#  Decoder (Generator) with Cross-Attention
# ═══════════════════════════════════════════════

class Decoder(nn.Module):
    """z(256) + condition(512) → 128×128×3.

    v2: Added cross-attention at 8×8 and 16×16 resolutions
    for fine-grained text-image alignment.
    """
    def __init__(self, latent_dim=256, condition_dim=512, base_ch=64):
        super().__init__()
        c = base_ch
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, c * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        # 4 → 8  → 16  → 32  → 64  → 128
        self.u1 = ResBlockUp(c * 8, c * 8, condition_dim)
        self.u2 = ResBlockUp(c * 8, c * 4, condition_dim)
        self.u3 = ResBlockUp(c * 4, c * 2, condition_dim)
        self.u4 = ResBlockUp(c * 2, c, condition_dim)
        self.u5 = ResBlockUp(c, c, condition_dim)

        # Cross-attention at 8×8 and 16×16 for text alignment
        self.cross_attn_8 = CrossAttention(c * 4, condition_dim, num_heads=4)
        self.cross_attn_16 = CrossAttention(c * 2, condition_dim, num_heads=4)

        self.tail = nn.Sequential(
            nn.Conv2d(c, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, condition):
        h = torch.cat([z, condition], dim=1)
        h = self.fc(h).view(-1, 512, 4, 4)
        h = self.u1(h, condition)                    # 4→8
        h = self.u2(h, condition)                    # 8→16
        h = self.cross_attn_8(h, condition)          # cross-attn at 8×8
        h = self.u3(h, condition)                    # 16→32
        h = self.cross_attn_16(h, condition)         # cross-attn at 16×16
        h = self.u4(h, condition)                    # 32→64
        h = self.u5(h, condition)                    # 64→128
        return self.tail(h)


# ═══════════════════════════════════════════════
#  Discriminator
# ═══════════════════════════════════════════════

class Discriminator(nn.Module):
    """128×128×3 + condition → real/fake logit.

    Uses projection discrimination: the condition embedding is dot-producted
    with the final feature map to produce a condition-aware score.
    """
    def __init__(self, base_ch=64, condition_dim=512):
        super().__init__()
        c = base_ch
        self.head = nn.utils.spectral_norm(nn.Conv2d(3, c, 3, 1, 1, bias=False))
        self.d1 = ResBlockDown(c, c * 2)           # 128→64
        self.d2 = ResBlockDown(c * 2, c * 4)       # 64→32
        self.d3 = ResBlockDown(c * 4, c * 8)       # 32→16
        self.d4 = ResBlockDown(c * 8, c * 8)       # 16→8
        self.d5 = ResBlockDown(c * 8, c * 8)       # 8→4

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
        self.fc_out = nn.utils.spectral_norm(
            nn.Linear(c * 8 * 4 * 4, 1))

        # Projection head: condition → feature-space embedding
        self.cond_proj = nn.utils.spectral_norm(
            nn.Linear(condition_dim, c * 8 * 4 * 4)
        )

    def forward(self, image, condition):
        h = self.act(self.head(image))
        h = self.d1(h)
        h = self.d2(h)
        h = self.d3(h)
        h = self.d4(h)
        h = self.d5(h)
        h = self.flatten(h)
        out = self.fc_out(h)

        # Projection with normalization to prevent explosion
        cond_emb = self.cond_proj(condition)
        feat_dim = h.shape[1]
        out = out + (h * cond_emb).sum(dim=1, keepdim=True) / (feat_dim ** 0.5)
        return out


# ═══════════════════════════════════════════════
#  Perceptual (VGG) Loss
# ═══════════════════════════════════════════════

class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss on relu2_2 and relu3_3."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([
            vgg[:9],    # relu2_2
            vgg[9:16],  # relu3_3
        ])
        for blk in self.blocks:
            for p in blk.parameters():
                p.requires_grad = False
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, fake, real):
        fake = (fake * 0.5 + 0.5)
        real = (real * 0.5 + 0.5)
        fake = (fake - self.mean) / self.std
        real = (real - self.mean) / self.std
        loss = 0.0
        for blk in self.blocks:
            fake = blk(fake)
            real = blk(real)
            loss += F.l1_loss(fake, real)
        return loss


# ═══════════════════════════════════════════════
#  VAE-GAN (top-level)
# ═══════════════════════════════════════════════

class VaeGan(nn.Module):
    """Conditional VAE-GAN for Text-to-Image (v2 with CLIP)."""
    def __init__(self, latent_dim=256, text_dim=512, condition_dim=512, base_ch=64):
        super().__init__()
        self.text_encoder = TextEncoder(text_dim, condition_dim)
        self.encoder = Encoder(base_ch, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, base_ch)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, text_embed):
        condition = self.text_encoder(text_embed)
        mu, logvar = self.encoder(image)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, condition)
        return recon, mu, logvar, condition

    @torch.no_grad()
    def generate(self, text_embed, num_images=None, temperature=1.0, device="cuda"):
        """Generate images from text description."""
        self.eval()
        condition = self.text_encoder(text_embed.to(device))
        if num_images is None:
            num_images = condition.shape[0]
        z = torch.randn(num_images, self.latent_dim, device=device) * temperature
        if condition.shape[0] == 1:
            condition = condition.expand(num_images, -1)
        elif condition.shape[0] != num_images:
            condition = condition[:num_images]
        images = self.decoder(z, condition)
        return images


# ═══════════════════════════════════════════════
#  Loss helpers
# ═══════════════════════════════════════════════

def vae_loss(recon, target, mu, logvar, kl_weight=0.0001):
    """L1 reconstruction + KL divergence."""
    recon_loss = F.l1_loss(recon, target)
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss


def matching_loss(condition, image_features):
    """Cosine similarity matching loss between text condition and image features.

    Forces the generated image features to align with the text condition,
    preventing the model from ignoring text input.
    """
    condition_norm = F.normalize(condition, dim=1)
    image_norm = F.normalize(image_features, dim=1)
    # Cosine similarity → want to maximize, so minimize negative
    return -torch.mean(torch.sum(condition_norm * image_norm, dim=1))


def d_loss_fn(real_logit, fake_logit):
    """Hinge loss for discriminator."""
    real_loss = F.relu(1.0 - real_logit).mean()
    fake_loss = F.relu(1.0 + fake_logit).mean()
    return real_loss + fake_loss


def g_loss_fn(fake_logit):
    """Hinge loss for generator (adversarial)."""
    return -fake_logit.mean()


def gradient_penalty(discriminator, real_images, fake_images, condition):
    """WGAN-GP gradient penalty."""
    batch = real_images.size(0)
    alpha = torch.rand(batch, 1, 1, 1, device=real_images.device)
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)

    d_interpolates = discriminator(interpolates, condition)
    grad_outputs = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
    )[0]

    gradients = gradients.view(batch, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
