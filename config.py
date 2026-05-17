"""Text-to-Image VAE-GAN Configuration"""
import os
import torch

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
GLOVE_DIR = os.path.join(BASE_DIR, "data", "glove")
GLOVE_PATH = os.path.join(GLOVE_DIR, "glove.6B.50d.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# --- Image ---
IMAGE_SIZE = 256             # 128→256 for sharper output on A100

# --- Model ---
LATENT_DIM = 512             # 256→512 more capacity for 256px
TEXT_EMBED_DIM = 50          # GloVe 50d
CONDITION_DIM = 512          # 256→512 stronger text conditioning
BASE_CHANNELS = 64           # multiplier for conv channels

# --- Training ---
BATCH_SIZE = 32              # 256px needs more memory per sample
EPOCHS = 300                 # more epochs for higher resolution

# Optimizer
LR_G = 5e-5                  # halved for fine-tuning stability
LR_D = 2e-4                  # halved for fine-tuning stability
BETA1 = 0.5
BETA2 = 0.999

# Loss weights
KL_WEIGHT = 0.0001           # KL divergence
ADV_WEIGHT = 1.0             # 0.5→1.0 stronger adversarial to fight blur
PERCEPTUAL_WEIGHT = 0.2      # 0.1→0.2 more perceptual detail
RECON_WEIGHT = 1.0           # L1 reconstruction
GP_WEIGHT = 1.5              # 2.0→1.5 less penalty, let D learn harder

# Gradient clipping
GRAD_CLIP = 1.0              # max gradient norm (prevents explosion)

# KL annealing
KL_ANNEAL_EPOCHS = 150       # 100→150 longer warmup for 300 epochs

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
