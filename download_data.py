"""
下载 Flowers 数据集的辅助文件 + GloVe 词向量
图片已放在 GAN/data/jpg/ 中
"""
import os
import sys
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
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
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


def prepare_flowers(data_dir):
    """下载辅助文件并生成文字描述"""
    os.makedirs(data_dir, exist_ok=True)

    # 下载数据集划分
    splits_path = os.path.join(data_dir, "setid.mat")
    if not os.path.exists(splits_path):
        print("下载数据集划分文件...")
        download_url("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat", splits_path)

    # 下载标签
    labels_path = os.path.join(data_dir, "imagelabels.mat")
    if not os.path.exists(labels_path):
        print("下载标签文件...")
        download_url("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat", labels_path)

    # 生成文字描述
    captions_json = os.path.join(data_dir, "captions.json")
    if not os.path.exists(captions_json):
        generate_captions(data_dir)


def generate_captions(data_dir):
    """根据花卉类别生成简单文字描述"""
    print("生成文字描述...")
    images_dir = os.path.join(data_dir, "jpg")
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

    # 读取标签
    labels_path = os.path.join(data_dir, "imagelabels.mat")
    labels = None
    if os.path.exists(labels_path):
        try:
            mat = sio.loadmat(labels_path)
            labels = mat['labels'].flatten()
        except Exception:
            pass

    # 花卉描述模板
    colors = ["red", "pink", "white", "yellow", "purple", "blue", "orange"]
    types = ["rose", "daisy", "tulip", "sunflower", "orchid", "lily"]

    captions_dict = {}
    for i, img_file in enumerate(image_files):
        if labels is not None and i < len(labels):
            label = int(labels[i])
            color = colors[label % len(colors)]
            flower = types[label % len(types)]
            desc = f"a {color} {flower}"
        else:
            desc = "a photo of a flower"
        captions_dict[img_file] = [desc]

    json_path = os.path.join(data_dir, "captions.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(captions_dict, f, ensure_ascii=False, indent=2)
    print(f"已生成 {json_path}，共 {len(captions_dict)} 张图片")


def download_glove(glove_dir):
    """下载 GloVe 6B 50d 词向量"""
    os.makedirs(glove_dir, exist_ok=True)
    glove_txt = os.path.join(glove_dir, "glove.6B.50d.txt")

    if os.path.exists(glove_txt):
        print(f"GloVe 词向量已存在: {glove_txt}")
        return

    print("下载 GloVe 6B 词向量 (~100MB)...")
    zip_path = os.path.join(glove_dir, "glove.6B.zip")
    download_url("https://nlp.stanford.edu/data/glove.6B.zip", zip_path)

    print("解压 GloVe...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(glove_dir)
    os.remove(zip_path)
    print(f"GloVe 已解压到 {glove_dir}")


def verify_data(data_dir, glove_path):
    print("\n=== 数据校验 ===")

    images_dir = os.path.join(data_dir, "jpg")
    if os.path.exists(images_dir):
        n_images = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        print(f"图片: {n_images} 张")
    else:
        print("图片目录不存在!")
        return False

    captions_json = os.path.join(data_dir, "captions.json")
    if os.path.exists(captions_json):
        with open(captions_json, 'r') as f:
            caps = json.load(f)
        print(f"文字描述: {len(caps)} 条")
    else:
        print("文字描述文件不存在!")
        return False

    if os.path.exists(glove_path):
        with open(glove_path, 'r', encoding='utf-8') as f:
            n_lines = sum(1 for _ in f)
        print(f"GloVe 词向量: {n_lines} 个词")
    else:
        print("GloVe 文件不存在!")
        return False

    print("校验通过!")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Text-to-Image cVAE - 数据准备")
    print("=" * 50)

    prepare_flowers(DATA_DIR)
    download_glove(GLOVE_DIR)
    verify_data(DATA_DIR, GLOVE_PATH)
