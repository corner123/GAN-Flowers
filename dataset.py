"""Flowers Dataset with CLIP text encoding."""
import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# CLIP text encoder (lazy loaded)
_clip_model = None


def get_clip_encoder():
    """Load CLIP ViT-B/32 model (singleton, loaded once)."""
    global _clip_model
    if _clip_model is None:
        import clip
        print("Loading CLIP ViT-B/32...")
        _clip_model, _ = clip.load("ViT-B/32", device="cpu")
        _clip_model.eval()
        print("CLIP loaded.")
    return _clip_model


def encode_text_clip(text):
    """Encode a single text string to CLIP embedding (512d)."""
    import clip
    model = get_clip_encoder()
    tokens = clip.tokenize([text], truncate=True)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    return text_features.squeeze(0).float()


def encode_text_clip_batch(texts):
    """Encode a batch of text strings to CLIP embeddings."""
    import clip
    model = get_clip_encoder()
    tokens = clip.tokenize(texts, truncate=True)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    return text_features.float()


class FlowersDataset(Dataset):
    """Oxford-102 Flowers with CLIP text encoding."""

    def __init__(self, data_dir, split="train",
                 image_size=128, augment=False, clip_cache_path=None):
        self.data_dir = data_dir
        self.image_size = image_size

        images_dir = os.path.join(data_dir, "jpg")
        captions_path = os.path.join(data_dir, "captions.json")

        with open(captions_path, 'r', encoding='utf-8') as f:
            all_captions = json.load(f)

        setid_path = os.path.join(data_dir, "setid.mat")
        if os.path.exists(setid_path):
            import scipy.io as sio
            mat = sio.loadmat(setid_path)
            if split == "train":
                ids = mat['trnid'].flatten()
            elif split == "val":
                ids = mat['valid'].flatten()
            else:
                ids = mat['tstid'].flatten()

            self.image_files = []
            for idx in ids:
                fname = f"image_{idx:05d}.jpg"
                if fname in all_captions:
                    self.image_files.append(fname)
        else:
            all_files = sorted(
                [f for f in os.listdir(images_dir) if f.endswith('.jpg')])
            random.seed(42)
            random.shuffle(all_files)
            n = len(all_files)
            if split == "train":
                self.image_files = all_files[:int(n * 0.8)]
            elif split == "val":
                self.image_files = all_files[int(n * 0.8):int(n * 0.9)]
            else:
                self.image_files = all_files[int(n * 0.9):]

        self.captions = all_captions
        self.images_dir = images_dir

        # Build transform
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        # Pre-encode all captions with CLIP (fast, avoids per-batch encoding)
        self.clip_cache_path = clip_cache_path or os.path.join(data_dir, "clip_embeddings.pt")
        self._precompute_clip_embeddings()

        print(f"[{split}] {len(self.image_files)} images "
              f"(augment={'on' if augment else 'off'})")

    def _precompute_clip_embeddings(self):
        """Pre-encode all captions with CLIP and cache to disk."""
        if os.path.exists(self.clip_cache_path):
            print(f"Loading CLIP cache: {self.clip_cache_path}")
            cached = torch.load(self.clip_cache_path, weights_only=True)
            self.clip_embeddings = cached
        else:
            print("Pre-encoding captions with CLIP (one-time)...")
            # Collect all unique captions
            all_captions_set = []
            for img_file in self.image_files:
                caps = self.captions.get(img_file, ["a flower"])
                all_captions_set.append(caps[0])  # use first caption

            # Batch encode
            batch_size = 256
            embeddings = []
            for i in range(0, len(all_captions_set), batch_size):
                batch = all_captions_set[i:i + batch_size]
                emb = encode_text_clip_batch(batch)
                embeddings.append(emb)
            self.clip_embeddings = torch.cat(embeddings, dim=0)

            # Cache to disk
            torch.save(self.clip_embeddings, self.clip_cache_path)
            print(f"CLIP cache saved: {self.clip_cache_path} "
                  f"({self.clip_embeddings.shape})")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        captions = self.captions.get(img_file, ["a flower"])
        caption = captions[0]

        # Use pre-computed CLIP embedding
        text_embed = self.clip_embeddings[idx]

        return image, text_embed, caption


if __name__ == "__main__":
    from config import DATA_DIR, IMAGE_SIZE

    ds = FlowersDataset(DATA_DIR, split="train", image_size=IMAGE_SIZE, augment=True)
    img, text, caption = ds[0]
    print(f"Image: {img.shape}, Text: {text.shape}, Caption: {caption}")
