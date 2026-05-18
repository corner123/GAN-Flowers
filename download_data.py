"""Download Oxford-102 Flowers dataset + GloVe word vectors."""
import os
import sys
import tarfile
import urllib.request
import zipfile
import scipy.io as sio
import json
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, GLOVE_DIR, GLOVE_PATH


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0'
    })
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                             desc=url.split('/')[-1]) as t:
        response = urllib.request.urlopen(req)
        total_size = int(response.headers.get('content-length', 0))
        t.total = total_size
        with open(output_path, 'wb') as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                t.update(len(chunk))


def download_images(data_dir):
    """Download Oxford-102 flower images and extract them."""
    images_dir = os.path.join(data_dir, "jpg")
    os.makedirs(images_dir, exist_ok=True)

    # Check if images already exist
    existing = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    if len(existing) >= 800:
        print(f"Images already exist: {len(existing)} files in {images_dir}")
        return

    # Download the 102 Category Flower Dataset images
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    tgz_path = os.path.join(data_dir, "102flowers.tgz")

    print(f"Downloading Oxford-102 Flowers images (~328 MB)...")
    download_url(url, tgz_path)

    print("Extracting images...")
    with tarfile.open(tgz_path, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.name.lower().endswith('.jpg')]
        for member in tqdm(members, desc="Extracting", unit="img"):
            # Rename: jpg/image_0001.jpg → jpg/image_00001.jpg (5-digit padding)
            basename = os.path.basename(member.name)
            # Original names like "image_00001.jpg"
            target_name = basename
            out_path = os.path.join(images_dir, target_name)
            with tar.extractfile(member) as src:
                with open(out_path, 'wb') as dst:
                    dst.write(src.read())

    os.remove(tgz_path)
    n = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    print(f"Extracted {n} images to {images_dir}")


def prepare_flowers(data_dir):
    """Download setid.mat, imagelabels.mat and generate captions.json."""
    os.makedirs(data_dir, exist_ok=True)

    # setid.mat (train/val/test split)
    splits_path = os.path.join(data_dir, "setid.mat")
    if not os.path.exists(splits_path):
        print("Downloading setid.mat...")
        download_url(
            "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
            splits_path)

    # imagelabels.mat
    labels_path = os.path.join(data_dir, "imagelabels.mat")
    if not os.path.exists(labels_path):
        print("Downloading imagelabels.mat...")
        download_url(
            "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
            labels_path)

    # captions.json — regenerate if missing or image count changed
    captions_json = os.path.join(data_dir, "captions.json")
    images_dir = os.path.join(data_dir, "jpg")
    n_images = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')]) if os.path.exists(images_dir) else 0
    if os.path.exists(captions_json):
        with open(captions_json, 'r') as f:
            n_caps = len(json.load(f))
        if n_caps != n_images:
            print(f"Captions count mismatch ({n_caps} vs {n_images} images), regenerating...")
            generate_captions(data_dir)
    else:
        generate_captions(data_dir)


def generate_captions(data_dir):
    """Generate diverse text captions from flower labels.

    Oxford-102 has 102 flower species. Each species gets multiple
    descriptive captions with varied structure for better CLIP encoding.
    """
    print("Generating captions...")
    images_dir = os.path.join(data_dir, "jpg")
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith('.jpg')])

    labels_path = os.path.join(data_dir, "imagelabels.mat")
    labels = None
    if os.path.exists(labels_path):
        try:
            mat = sio.loadmat(labels_path)
            labels = mat['labels'].flatten()
        except Exception:
            pass

    # Oxford-102 flower species names (102 categories)
    flower_names = [
        "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
        "sweet pea", "english marigold", "tiger lily", "moon orchid",
        "bird of paradise", "monkshood", "globe thistle", "snapdragon",
        "colt's foot", "king protea", "spear thistle", "yellow iris",
        "globe flower", "purple coneflower", "peruvian lily", "balloon flower",
        "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
        "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
        "stemless gentian", "artichoke", "sweet william", "carnation",
        "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
        "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
        "lenten rose", "barbeton daisy", "daffodil", "sword lily",
        "poinsettia", "bolero deep blue", "wallflower", "marigold",
        "buttercup", "oxeye daisy", "common dandelion", "petunia",
        "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff",
        "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
        "cautleya spicata", "japanese anemone", "black-eyed susan",
        "silverbush", "californian poppy", "osteospermum", "spring crocus",
        "bearded iris", "windflower", "tree poppy", "gazania", "azalea",
        "water lily", "rose", "thorn apple", "morning glory", "passion flower",
        "lotus", "toad lily", "anthurium", "frangipani", "clematis",
        "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
        "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm",
        "ball moss", "foxglove", "bougainvillea", "camellia", "mallow",
        "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
        "blackberry lily",
    ]

    # Caption templates for diversity
    templates = [
        "a photo of a {}",
        "a {} flower",
        "a beautiful {}",
        "a close-up of a {}",
        "a blooming {}",
        "a {} in a garden",
    ]

    captions_dict = {}
    for i, img_file in enumerate(image_files):
        if labels is not None and i < len(labels):
            label = int(labels[i]) - 1  # labels are 1-indexed
            if 0 <= label < len(flower_names):
                flower = flower_names[label]
            else:
                flower = "flower"
        else:
            flower = "flower"

        # Generate multiple diverse captions for each image
        caps = []
        for template in templates[:3]:  # use 3 templates per image
            caps.append(template.format(flower))
        captions_dict[img_file] = caps

    json_path = os.path.join(data_dir, "captions.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(captions_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved {json_path} ({len(captions_dict)} images, 102 species)")


def download_glove(glove_dir):
    """Download GloVe 6B 50d (~170 MB)."""
    os.makedirs(glove_dir, exist_ok=True)
    glove_txt = os.path.join(glove_dir, "glove.6B.50d.txt")

    if os.path.exists(glove_txt):
        print(f"GloVe already exists: {glove_txt}")
        return

    print("Downloading GloVe 6B (~100 MB zip)...")
    zip_path = os.path.join(glove_dir, "glove.6B.zip")
    download_url("https://nlp.stanford.edu/data/glove.6B.zip", zip_path)

    print("Extracting GloVe...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(glove_dir)
    os.remove(zip_path)
    print(f"GloVe extracted to {glove_dir}")


def verify_data(data_dir, glove_path):
    print("\n=== Data verification ===")

    images_dir = os.path.join(data_dir, "jpg")
    if os.path.exists(images_dir):
        n = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        print(f"Images: {n}")
    else:
        print("Images dir missing!")
        return False

    captions_json = os.path.join(data_dir, "captions.json")
    if os.path.exists(captions_json):
        with open(captions_json, 'r') as f:
            caps = json.load(f)
        print(f"Captions: {len(caps)}")
    else:
        print("Captions file missing!")
        return False

    if os.path.exists(glove_path):
        with open(glove_path, 'r', encoding='utf-8') as f:
            n = sum(1 for _ in f)
        print(f"GloVe: {n} words")
    else:
        print("GloVe file missing!")
        return False

    print("All OK!")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("VAE-GAN Text-to-Image — Data Preparation")
    print("=" * 50)

    download_images(DATA_DIR)
    prepare_flowers(DATA_DIR)
    download_glove(GLOVE_DIR)
    verify_data(DATA_DIR, GLOVE_PATH)
