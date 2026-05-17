"""Flowers Dataset with data augmentation and GloVe text encoding."""
import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class GloVeEmbedding:
    """GloVe word vector loader."""
    def __init__(self, glove_path, dim=50):
        self.dim = dim
        self.word2vec = {}
        self.mean_vec = None

        print(f"Loading GloVe: {glove_path}")
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                if len(vec) == dim:
                    self.word2vec[word] = vec

        all_vecs = np.stack(list(self.word2vec.values()))
        self.mean_vec = all_vecs.mean(axis=0)
        print(f"Loaded {len(self.word2vec)} word vectors, dim={dim}")

    def encode_sentence(self, sentence, max_words=20):
        words = sentence.lower().strip().split()[:max_words]
        vecs = [self.word2vec[w] for w in words if w in self.word2vec]
        if not vecs:
            return self.mean_vec.copy()
        return np.mean(vecs, axis=0).astype(np.float32)


class FlowersDataset(Dataset):
    """Oxford-102 Flowers with optional data augmentation."""

    def __init__(self, data_dir, glove_path, split="train",
                 image_size=128, glove_dim=50, augment=False, glove=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.glove = glove if glove is not None else GloVeEmbedding(glove_path, glove_dim)

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
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        print(f"[{split}] {len(self.image_files)} images "
              f"(augment={'on' if augment else 'off'})")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        captions = self.captions.get(img_file, ["a flower"])
        caption = random.choice(captions)
        text_embed = torch.from_numpy(self.glove.encode_sentence(caption))

        return image, text_embed, caption


def load_glove_for_inference(glove_path, dim=50):
    return GloVeEmbedding(glove_path, dim)


if __name__ == "__main__":
    from config import DATA_DIR, GLOVE_PATH, IMAGE_SIZE

    ds = FlowersDataset(DATA_DIR, GLOVE_PATH, split="train",
                        image_size=IMAGE_SIZE, augment=True)
    img, text, caption = ds[0]
    print(f"Image: {img.shape}, Text: {text.shape}, Caption: {caption}")
