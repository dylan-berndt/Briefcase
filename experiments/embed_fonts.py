"""
Generates one visual embedding per font using the pretrained font ViT
checkpoint (checkpoints/pretrain/best). Only rasterizes lowercase a-z
(what generateEmbeddings actually averages over) instead of the full
~200-character set collectFontSetPaths normally produces, to keep disk
usage bounded -- the sentinel-file check in imagesFromFont then makes
collectFontSetPaths treat these fonts as already rasterized.
"""
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from utils.loaders.standard import imagesFromFont, collectFontSetPaths
from utils.loaders.myfonts import loadMyFontsImagePaths
from utils.vit import ViT
from utils.embeddings import generateEmbeddings

FONT_SIZE = 32
IMAGE_SIZE = int(FONT_SIZE * 1.5)
LOWERCASE_LATIN = [chr(c) for c in range(ord('a'), ord('z') + 1)]

SOURCE_DIRS = ["google", "dafont"]
MYFONTS_DIR = "dataset"
CHECKPOINT_DIR = os.path.join("checkpoints", "pretrain", "best")
OUTPUT_NAME = "font_vit_embeddings"


def rasterize(directory):
    fontsGlob = os.path.join(directory, "fonts", "**", "*")
    paths = glob(fontsGlob + ".ttf", recursive=True) + glob(fontsGlob + ".otf", recursive=True)
    os.makedirs(os.path.join(directory, "bitmaps"), exist_ok=True)
    os.makedirs(os.path.join(directory, "sdf"), exist_ok=True)

    for i, path in enumerate(paths):
        if not os.path.isfile(path):
            continue
        try:
            imagesFromFont(path, FONT_SIZE, IMAGE_SIZE, save=directory, chars=LOWERCASE_LATIN)
        except Exception as e:
            print(path, e)
        print(f"\r[{directory}] Fonts rasterized: {i + 1}/{len(paths)}", end="")
    print()


def main():
    for directory in SOURCE_DIRS:
        if not os.path.isdir(os.path.join(directory, "fonts")):
            print(f"Skipping {directory}: no fonts/ directory found (did setup_data.sh run?)")
            continue
        rasterize(directory)

    names, letters, paths = [], [], []
    for directory in SOURCE_DIRS:
        if not os.path.isdir(os.path.join(directory, "fonts")):
            continue
        data = collectFontSetPaths(directory, FONT_SIZE, "bitmaps")
        names.append(data["names"]); letters.append(data["letters"]); paths.append(data["paths"])

    if os.path.isdir(os.path.join(MYFONTS_DIR, "fontimage")):
        data = loadMyFontsImagePaths(MYFONTS_DIR, FONT_SIZE)
        names.append(data["names"]); letters.append(data["letters"]); paths.append(data["paths"])
    else:
        print(f"Skipping {MYFONTS_DIR}: no fontimage/ directory found")

    fontData = {
        "names": np.concatenate(names, axis=0),
        "letters": np.concatenate(letters, axis=0),
        "paths": np.concatenate(paths, axis=0),
    }

    print(f"Collected {len(np.unique(fontData['names']))} unique fonts, {len(fontData['paths'])} glyph images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = ViT.load(CHECKPOINT_DIR)
    model = model.to(device)
    model.eval()

    embeddings = generateEmbeddings(fontData, model, fileName=OUTPUT_NAME)
    print(f"Saved {len(embeddings)} font embeddings to embeddings/{OUTPUT_NAME}.json")


if __name__ == "__main__":
    main()
