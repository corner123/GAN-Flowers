"""Text-to-Image VAE-GAN Configuration (v2 — CLIP)"""
import os
import torch

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CLIP_CACHE = os.path.join(DATA_DIR, "clip_embeddings.pt")

# --- Image ---
IMAGE_SIZE = 128

# --- Model ---
LATENT_DIM = 256
TEXT_EMBED_DIM = 512         # CLIP ViT-B/32 output dim (replaces GloVe 50d)
CONDITION_DIM = 512          # must match TEXT_EMBED_DIM
BASE_CHANNELS = 64           # multiplier for conv channels

# --- Training ---
BATCH_SIZE = 64
EPOCHS = 300

# Optimizer
LR_G = 1e-4                  # Generator (VAE) learning rate
LR_D = 4e-4                  # Discriminator learning rate
BETA1 = 0.5
BETA2 = 0.999

# Loss weights
KL_WEIGHT = 0.0001           # KL divergence
ADV_WEIGHT = 1.0             # 0.5→1.0 stronger adversarial to fight blur
PERCEPTUAL_WEIGHT = 0.1      # Perceptual (VGG) loss
RECON_WEIGHT = 1.0           # L1 reconstruction
GP_WEIGHT = 1.5              # 2.0→1.5 less penalty, let D learn harder
MATCH_WEIGHT = 0.5           # text-image matching loss (new)

# Gradient clipping
GRAD_CLIP = 1.0              # max gradient norm (prevents explosion)

# KL annealing
KL_ANNEAL_EPOCHS = 100       # linearly increase KL weight over this many epochs

# Mixed precision
USE_AMP = True               # automatic mixed precision (A100 has great FP16 perf)

# Logging
LOG_INTERVAL = 20            # batches between progress prints
SAVE_INTERVAL = 20           # epochs between checkpoint saves
SAMPLE_INTERVAL = 10         # epochs between saving sample images
N_SAMPLES = 8                # number of samples to save

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
