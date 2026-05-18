"""Text-to-Image inference with VAE-GAN (v2 — CLIP)."""
import os
import sys
import argparse

import torch
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (IMAGE_SIZE, LATENT_DIM, TEXT_EMBED_DIM,
                    CONDITION_DIM, BASE_CHANNELS, DEVICE,
                    CHECKPOINT_DIR, OUTPUT_DIR)
from model import VaeGan
from dataset import encode_text_clip


def generate_from_text(text, model_path=None, num_images=4, temperature=1.0, seed=None):
    """Generate images from a text description using CLIP encoding."""
    if seed is not None:
        torch.manual_seed(seed)

    print(f"Text: \"{text}\"")
    print("Encoding text with CLIP...")
    text_embed = encode_text_clip(text).unsqueeze(0)
    print(f"Embedding: {text_embed.shape}")

    model = VaeGan(latent_dim=LATENT_DIM, text_dim=TEXT_EMBED_DIM,
                   condition_dim=CONDITION_DIM, base_ch=BASE_CHANNELS)

    if model_path is None:
        model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
        state_dict = ckpt.get('vae_gan_state_dict', ckpt.get('model_state_dict', ckpt))
        model.load_state_dict(state_dict)
        epoch = ckpt.get('epoch', '?')
        val_loss = ckpt.get('val_loss', '?')
        print(f"Model loaded: {model_path} (epoch={epoch}, val_loss={val_loss})")
    else:
        print(f"WARNING: No model at {model_path} — using random weights!")

    model.to(DEVICE)
    model.eval()

    text_embeds = text_embed.expand(num_images, -1).to(DEVICE)
    with torch.no_grad():
        generated = model.generate(text_embeds, num_images=num_images,
                                   temperature=temperature, device=DEVICE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_text = text.replace(' ', '_')[:40]
    output_path = os.path.join(OUTPUT_DIR, f"generated_{safe_text}.png")
    save_image(generated * 0.5 + 0.5, output_path, nrow=min(num_images, 4))
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="VAE-GAN Text-to-Image (CLIP)")
    parser.add_argument("--text", type=str, default="a red rose",
                        help="Text description")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path (default: checkpoints/best_model.pt)")
    parser.add_argument("--num", type=int, default=4,
                        help="Number of images to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default 1.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    args = parser.parse_args()

    generate_from_text(args.text, args.model, args.num, args.temperature, args.seed)


if __name__ == "__main__":
    main()
