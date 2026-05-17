"""VAE-GAN for Text-to-Image Generation.

Architecture:
  TextEncoder:  GloVe 50d → condition vector (256d)
  Encoder:      128x128x3 → μ, logσ² (latent_dim=256)
  Decoder:      z(256) + condition(256) → 128×128×3   (generator)
  Discriminator: 128×128×3 + condition(256) → real/fake logit

Losses:
  - L1 reconstruction
  - Perceptual (VGG16 feature matching)
  - KL divergence (annealed)
  - Adversarial (hinge loss + gradient penalty)
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
#  Text Encoder
# ═══════════════════════════════════════════════

class TextEncoder(nn.Module):
    """GloVe 50d → condition vector."""
    def __init__(self, text_dim=50, hidden_dim=512, condition_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, condition_dim),
        )

    def forward(self, text_embed):
        return self.net(text_embed)


# ═══════════════════════════════════════════════
#  Image Encoder
# ═══════════════════════════════════════════════

class Encoder(nn.Module):
    """Image → μ, logσ² (6 downsamples → 4×4)."""
    def __init__(self, base_ch=64, latent_dim=256):
        super().__init__()
        c = base_ch
        self.head = nn.Conv2d(3, c, 3, 1, 1, bias=False)
        # 256 → 128 → 64 → 32 → 16 → 8 → 4
        self.d1 = ResBlockDown(c, c * 2)     # 256 → 128
        self.d2 = ResBlockDown(c * 2, c * 4) # 128 → 64
        self.d3 = ResBlockDown(c * 4, c * 8) # 64  → 32
        self.d4 = ResBlockDown(c * 8, c * 8) # 32  → 16
        self.d5 = ResBlockDown(c * 8, c * 8) # 16  → 8
        self.d6 = ResBlockDown(c * 8, c * 8) # 8   → 4
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
        h = self.d6(h)
        h = self.flatten(h)
        return self.fc_mu(h), self.fc_logvar(h)


# ═══════════════════════════════════════════════
#  Decoder (Generator)
# ═══════════════════════════════════════════════

class Decoder(nn.Module):
    """z + condition → 256×256×3 (6 upsamples from 4×4)."""
    def __init__(self, latent_dim=256, condition_dim=256, base_ch=64):
        super().__init__()
        c = base_ch
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, c * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        # 4 → 8  → 16  → 32  → 64  → 128 → 256
        self.u1 = ResBlockUp(c * 8, c * 8, condition_dim)
        self.u2 = ResBlockUp(c * 8, c * 4, condition_dim)
        self.u3 = ResBlockUp(c * 4, c * 2, condition_dim)
        self.u4 = ResBlockUp(c * 2, c, condition_dim)
        self.u5 = ResBlockUp(c, c, condition_dim)
        self.u6 = ResBlockUp(c, c, condition_dim)
        self.tail = nn.Sequential(
            nn.Conv2d(c, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, condition):
        h = torch.cat([z, condition], dim=1)
        h = self.fc(h).view(-1, 512, 4, 4)
        h = self.u1(h, condition)
        h = self.u2(h, condition)
        h = self.u3(h, condition)
        h = self.u4(h, condition)
        h = self.u5(h, condition)
        h = self.u6(h, condition)
        return self.tail(h)


# ═══════════════════════════════════════════════
#  Discriminator
# ═══════════════════════════════════════════════

class Discriminator(nn.Module):
    """256×256×3 + condition → real/fake logit (6 downsample layers).

    Uses projection discrimination: the condition embedding is dot-producted
    with the final feature map to produce a condition-aware score.
    """
    def __init__(self, base_ch=64, condition_dim=256):
        super().__init__()
        c = base_ch
        self.head = nn.utils.spectral_norm(nn.Conv2d(3, c, 3, 1, 1, bias=False))
        self.d1 = ResBlockDown(c, c * 2)           # 256→128
        self.d2 = ResBlockDown(c * 2, c * 4)       # 128→64
        self.d3 = ResBlockDown(c * 4, c * 8)       # 64→32
        self.d4 = ResBlockDown(c * 8, c * 8)       # 32→16
        self.d5 = ResBlockDown(c * 8, c * 8)       # 16→8
        self.d6 = ResBlockDown(c * 8, c * 8)       # 8→4

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
        h = self.d6(h)
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
        # VGG expects ImageNet-normalized inputs
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, fake, real):
        # Map from [-1, 1] to [0, 1] for VGG normalization
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
    """Conditional VAE-GAN for Text-to-Image."""
    def __init__(self, latent_dim=256, text_dim=50, condition_dim=256, base_ch=64):
        super().__init__()
        self.text_encoder = TextEncoder(text_dim, condition_dim=condition_dim)
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
        """Generate images from text description.

        Args:
            text_embed: (N, text_dim) or (1, text_dim)
            num_images: how many images to generate (default: len(text_embed))
            temperature: scaling factor for latent sampling std (1.0=default, >1=more diverse)
            device: target device
        """
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


def d_loss_fn(real_logit, fake_logit):
    """Hinge loss for discriminator."""
    real_loss = F.relu(1.0 - real_logit).mean()
    fake_loss = F.relu(1.0 + fake_logit).mean()
    return real_loss + fake_loss


def g_loss_fn(fake_logit):
    """Hinge loss for generator (adversarial)."""
    return -fake_logit.mean()


def gradient_penalty(discriminator, real_images, fake_images, condition):
    """WGAN-GP gradient penalty (unconditional on condition for simplicity)."""
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
