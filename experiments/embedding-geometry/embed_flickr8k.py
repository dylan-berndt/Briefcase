"""
Embeds Flickr8k images with DINOv2 (self-supervised, never saw text) and
its captions with BGE-large-en-v1.5 (the same text encoder used throughout
experiments/diffusion-searches/ for fonts, never finetuned on this task).
See README.md for why these specific encoders were chosen.

    python3 experiments/embedding-geometry/embed_flickr8k.py
"""
import os
import pickle

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "flickr8k")
IMAGE_DIR = os.path.join(DATA_DIR, "Flicker8k_Dataset")  # sic -- the dataset's own folder name
IMAGE_EMB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_embeddings.pkl")
TEXT_EMB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "text_embeddings.pkl")


def findTokenFile():
    direct = os.path.join(DATA_DIR, "Flickr8k.token.txt")
    if os.path.exists(direct):
        return direct
    for root, _, files in os.walk(DATA_DIR):
        if "Flickr8k.token.txt" in files:
            return os.path.join(root, "Flickr8k.token.txt")
    raise FileNotFoundError("Flickr8k.token.txt not found -- run download_flickr8k.py first")


def loadCaptions():
    """Returns {imageFilename: [caption1, ..., caption5]}."""
    path = findTokenFile()
    captions = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, caption = line.split("\t")
            imageName = key.split("#")[0]
            captions.setdefault(imageName, []).append(caption)
    return captions


def embedImages(imageNames, device):
    print(f"loading DINOv2 (self-supervised, never trained with text)...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = {}
    batch, batchNames = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            out = model(x).cpu().numpy()
        for n, vec in zip(batchNames, out):
            embeddings[n] = vec
        batch.clear()
        batchNames.clear()

    for i, name in enumerate(imageNames):
        path = os.path.join(IMAGE_DIR, name)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"skip {name}: {e}")
            continue
        batch.append(preprocess(img))
        batchNames.append(name)
        if len(batch) >= 64:
            flush()
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(imageNames)}")
    flush()
    return embeddings


def embedCaptions(captions, device):
    print("loading BGE-large-en-v1.5 (generic sentence embedder, never finetuned on image-caption retrieval)...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

    allCaptions = []
    index = []  # (imageName, captionIdx)
    for name, caps in captions.items():
        for i, c in enumerate(caps):
            allCaptions.append(c)
            index.append((name, i))

    print(f"embedding {len(allCaptions)} captions...")
    vectors = model.encode(allCaptions, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    perImage = {}
    for (name, i), vec in zip(index, vectors):
        perImage.setdefault(name, []).append(vec)
    return {name: np.stack(vecs) for name, vecs in perImage.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    captions = loadCaptions()
    imageNames = sorted(captions.keys())
    print(f"{len(imageNames)} images, {sum(len(v) for v in captions.values())} captions total")

    if os.path.exists(IMAGE_EMB_PATH):
        print(f"{IMAGE_EMB_PATH} already exists, skipping image embedding")
    else:
        imageEmbeddings = embedImages(imageNames, device)
        with open(IMAGE_EMB_PATH, "wb") as f:
            pickle.dump(imageEmbeddings, f)
        print(f"saved {len(imageEmbeddings)} image embeddings to {IMAGE_EMB_PATH}")

    if os.path.exists(TEXT_EMB_PATH):
        print(f"{TEXT_EMB_PATH} already exists, skipping text embedding")
    else:
        textEmbeddings = embedCaptions(captions, device)
        with open(TEXT_EMB_PATH, "wb") as f:
            pickle.dump(textEmbeddings, f)
        print(f"saved {len(textEmbeddings)} fonts' worth of caption embeddings to {TEXT_EMB_PATH}")


if __name__ == "__main__":
    main()
