"""Train VAE-GAN for Text-to-Image Generation on A100 GPU."""
import os
import sys
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (DATA_DIR, GLOVE_PATH, IMAGE_SIZE, LATENT_DIM, TEXT_EMBED_DIM,
                    CONDITION_DIM, BASE_CHANNELS, BATCH_SIZE, EPOCHS,
                    LR_G, LR_D, BETA1, BETA2, KL_WEIGHT, ADV_WEIGHT,
                    PERCEPTUAL_WEIGHT, RECON_WEIGHT, GP_WEIGHT,
                    KL_ANNEAL_EPOCHS, USE_AMP, LOG_INTERVAL,
                    SAVE_INTERVAL, SAMPLE_INTERVAL, N_SAMPLES,
                    DEVICE, OUTPUT_DIR, CHECKPOINT_DIR)
from dataset import FlowersDataset
from model import (VaeGan, Discriminator, PerceptualLoss,
                   vae_loss, d_loss_fn, g_loss_fn, gradient_penalty)


def save_sample_images(vae_gan, dataset, epoch, device, n=N_SAMPLES):
    """Save: top row = real, bottom row = reconstruction."""
    vae_gan.eval()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images, text_embeds = [], []
    for i in range(min(n, len(dataset))):
        img, te, _ = dataset[i]
        images.append(img)
        text_embeds.append(te)

    images = torch.stack(images).to(device)
    text_embeds = torch.stack(text_embeds).to(device)

    with torch.no_grad():
        recon, _, _, _ = vae_gan(images, text_embeds)

    comparison = torch.cat([images[:n], recon[:n]])
    save_image(comparison * 0.5 + 0.5,
               os.path.join(OUTPUT_DIR, f"recon_epoch_{epoch:03d}.png"),
               nrow=n)
    print(f"  [Sample] saved recon_epoch_{epoch:03d}.png")


def save_generated(vae_gan, dataset, epoch, device, n=4):
    """Save text-to-image generation samples."""
    vae_gan.eval()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    text_embeds, captions = [], []
    for i in range(min(n, len(dataset))):
        _, te, cap = dataset[i]
        text_embeds.append(te)
        captions.append(cap)

    text_embeds = torch.stack(text_embeds).to(device)
    with torch.no_grad():
        gen = vae_gan.generate(text_embeds, num_images=len(text_embeds), device=device)

    save_image(gen * 0.5 + 0.5,
               os.path.join(OUTPUT_DIR, f"generated_epoch_{epoch:03d}.png"),
               nrow=min(n, len(gen)))
    print(f"  [Sample] saved generated_epoch_{epoch:03d}.png")


def train(args):
    print("=" * 60)
    print("VAE-GAN Text-to-Image Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"AMP: {USE_AMP}")
    print()

    # --- Data ---
    train_dataset = FlowersDataset(
        DATA_DIR, GLOVE_PATH, split="train",
        image_size=IMAGE_SIZE, glove_dim=TEXT_EMBED_DIM, augment=True)

    val_dataset = FlowersDataset(
        DATA_DIR, GLOVE_PATH, split="val",
        image_size=IMAGE_SIZE, glove_dim=TEXT_EMBED_DIM, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print()

    # --- Models ---
    vae_gan = VaeGan(latent_dim=LATENT_DIM, text_dim=TEXT_EMBED_DIM,
                     condition_dim=CONDITION_DIM, base_ch=BASE_CHANNELS).to(DEVICE)
    discriminator = Discriminator(base_ch=BASE_CHANNELS, condition_dim=CONDITION_DIM).to(DEVICE)
    perceptual = PerceptualLoss().to(DEVICE)

    n_g_params = sum(p.numel() for p in vae_gan.parameters())
    n_d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator params:   {n_g_params:,} ({n_g_params/1e6:.2f}M)")
    print(f"Discriminator params: {n_d_params:,} ({n_d_params/1e6:.2f}M)")
    print(f"Perceptual loss: VGG16 (frozen)")
    print()

    # --- Optimizers ---
    opt_g = torch.optim.Adam(vae_gan.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    # --- Schedulers ---
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=EPOCHS)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=EPOCHS)

    # --- AMP ---
    scaler_g = torch.amp.GradScaler('cuda') if USE_AMP and DEVICE == "cuda" else None
    scaler_d = torch.amp.GradScaler('cuda') if USE_AMP and DEVICE == "cuda" else None

    # --- Directories ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_val_loss = float('inf')
    total_steps = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Step schedulers at start of epoch (after previous epoch's optimizer steps)
        if epoch > 1:
            sched_g.step()
            sched_d.step()

        # KL annealing weight (linear warmup)
        kl_w = KL_WEIGHT * min(1.0, epoch / max(1, KL_ANNEAL_EPOCHS))

        vae_gan.train()
        discriminator.train()

        d_loss_sum = g_loss_sum = recon_sum = kl_sum = percep_sum = adv_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS}", unit="batch")
        for batch_idx, (images, text_embeds, _) in enumerate(pbar):
            images = images.to(DEVICE, non_blocking=True)
            text_embeds = text_embeds.to(DEVICE, non_blocking=True)
            batch_size_actual = images.size(0)
            total_steps += 1

            # ═══════════════════════════════════
            #  1. Train Discriminator
            # ═══════════════════════════════════
            with torch.no_grad():
                recon, _, _, condition = vae_gan(images, text_embeds)

            # Only train D every other step sometimes (optional balancing)
            n_critic = 1
            for _ in range(n_critic):
                opt_d.zero_grad(set_to_none=True)

                if scaler_d is not None:
                    with torch.amp.autocast('cuda'):
                        real_logit = discriminator(images, condition)
                        fake_logit = discriminator(recon.detach(), condition)
                        d_loss = d_loss_fn(real_logit, fake_logit)
                        gp = gradient_penalty(discriminator, images, recon.detach(), condition)
                        d_total = d_loss + GP_WEIGHT * gp
                    scaler_d.scale(d_total).backward()
                    scaler_d.step(opt_d)
                    scaler_d.update()
                else:
                    real_logit = discriminator(images, condition)
                    fake_logit = discriminator(recon.detach(), condition)
                    d_loss = d_loss_fn(real_logit, fake_logit)
                    gp = gradient_penalty(discriminator, images, recon.detach(), condition)
                    (d_loss + GP_WEIGHT * gp).backward()
                    opt_d.step()

            # ═══════════════════════════════════
            #  2. Train Generator (VAE)
            # ═══════════════════════════════════
            opt_g.zero_grad(set_to_none=True)

            if scaler_g is not None:
                with torch.amp.autocast('cuda'):
                    recon, mu, logvar, condition = vae_gan(images, text_embeds)
                    # Reconstruction losses
                    l1_loss, kl_loss = vae_loss(recon, images, mu, logvar, kl_w)
                    p_loss = perceptual(recon, images)
                    # Adversarial loss
                    fake_logit_g = discriminator(recon, condition)
                    adv_loss = g_loss_fn(fake_logit_g)
                    # Total
                    g_total = (RECON_WEIGHT * l1_loss + kl_loss +
                               PERCEPTUAL_WEIGHT * p_loss + ADV_WEIGHT * adv_loss)
                scaler_g.scale(g_total).backward()
                scaler_g.step(opt_g)
                scaler_g.update()
            else:
                recon, mu, logvar, condition = vae_gan(images, text_embeds)
                l1_loss, kl_loss = vae_loss(recon, images, mu, logvar, kl_w)
                p_loss = perceptual(recon, images)
                fake_logit_g = discriminator(recon, condition)
                adv_loss = g_loss_fn(fake_logit_g)
                g_total = (RECON_WEIGHT * l1_loss + kl_loss +
                           PERCEPTUAL_WEIGHT * p_loss + ADV_WEIGHT * adv_loss)
                g_total.backward()
                opt_g.step()

            # Track losses
            d_loss_sum += d_loss.item()
            g_loss_sum += g_total.item()
            recon_sum += l1_loss.item()
            kl_sum += kl_loss.item()
            percep_sum += p_loss.item()
            adv_sum += adv_loss.item()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                pbar.set_postfix({
                    'D': f'{d_loss.item():.3f}',
                    'G': f'{g_total.item():.3f}',
                    'L1': f'{l1_loss.item():.3f}',
                    'KL': f'{kl_loss.item():.4f}',
                    'P': f'{p_loss.item():.3f}',
                    'Adv': f'{adv_loss.item():.3f}',
                    'kl_w': f'{kl_w:.1e}',
                })

        # --- End of epoch ---
        n_batches = len(train_loader)
        elapsed = time.time() - epoch_start
        lr_g = sched_g.get_last_lr()[0]
        lr_d = sched_d.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"D={d_loss_sum/n_batches:.3f} G={g_loss_sum/n_batches:.3f} "
              f"L1={recon_sum/n_batches:.4f} KL={kl_sum/n_batches:.5f} "
              f"P={percep_sum/n_batches:.3f} Adv={adv_sum/n_batches:.3f} "
              f"lr_g={lr_g:.1e} lr_d={lr_d:.1e} | {elapsed:.0f}s")

        # --- Validation ---
        vae_gan.eval()
        val_total = 0.0
        n_val = 0
        with torch.no_grad():
            for images, text_embeds, _ in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                text_embeds = text_embeds.to(DEVICE, non_blocking=True)
                recon, mu, logvar, _ = vae_gan(images, text_embeds)
                l1, kl = vae_loss(recon, images, mu, logvar, kl_weight=kl_w)
                val_total += (l1 + kl).item()
                n_val += 1
        val_loss = val_total / max(n_val, 1)
        print(f"  Val: {val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'vae_gan_state_dict': vae_gan.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  [Save] Best model (val_loss={val_loss:.4f})")

        # Sample images
        if epoch % SAMPLE_INTERVAL == 0 or epoch == 1:
            save_sample_images(vae_gan, train_dataset, epoch, DEVICE)
            save_generated(vae_gan, train_dataset, epoch, DEVICE)

        # Checkpoints
        if epoch % SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'vae_gan_state_dict': vae_gan.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(CHECKPOINT_DIR, f"vaegan_epoch_{epoch:03d}.pt"))
            print(f"  [Save] Checkpoint epoch {epoch}")

    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Model: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr_g", type=float, default=None)
    parser.add_argument("--lr_d", type=float, default=None)
    args = parser.parse_args()

    import config
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr_g:
        config.LR_G = args.lr_g
    if args.lr_d:
        config.LR_D = args.lr_d

    train(args)
