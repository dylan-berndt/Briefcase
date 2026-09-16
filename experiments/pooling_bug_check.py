"""
Concrete demonstration that generateEmbeddings' per-font vector (normalize
each of the 26 letter CLS embeddings, then mean-pool -- with no final
renormalization, utils/embeddings.py:51) is NOT the same quantity as the
average pairwise cosine similarity between two fonts' letters, and does
not itself lie on the unit hypersphere.

For unit vectors u_i (font A's 26 letters) and v_j (font B's 26 letters):

    mean_{i,j} cos(u_i, v_j) = mean_{i,j} u_i . v_j = mean(u_i) . mean(v_j)
                             = ubar . vbar            (plain dot product)

    cos(ubar, vbar) = (ubar . vbar) / (|ubar| |vbar|)

These are equal only when |ubar| = |vbar| = 1, i.e. only if every font's
26 letters already point in exactly the same direction. Since |ubar| < 1
in general (it's a norm of an average of unit vectors, not a unit vector
itself), and varies font-to-font with how much the 26 letters agree in
direction, dividing by |ubar||vbar| does not recover the true average
pairwise cosine -- it rescales it by a font-specific, letter-consistency
-dependent factor.

    python3 experiments/pooling_bug_check.py
"""
import os
import random
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from utils.pretraining import characters, loadImage
from utils.vit import ViT

CHECKPOINT_DIR = os.path.join("checkpoints", "pretrain", "best")
LOWERCASE_LATIN = [chr(c) for c in range(ord('a'), ord('z') + 1)]

SOURCES = [
    (os.path.join("google", "bitmaps"), ".bmp"),
    (os.path.join("dafont", "bitmaps"), ".bmp"),
    (os.path.join("dataset", "smallimage"), ".bmp"),
]


def collectExistingPaths(directory, ext):
    paths = glob(os.path.join(directory, f"*{ext}"))
    byName = {}
    for path in paths:
        stem = os.path.basename(path)[:-len(ext)]
        if len(stem) < 3 or "ԵՒ" in stem:
            continue
        char = stem[-2]
        if char not in characters:
            continue
        byName.setdefault(stem[:-3], {})[char] = path
    return byName


def letterEmbeddings(model, device, letterPaths):
    images = [torch.tensor(loadImage(letterPaths[l])[1], dtype=torch.float32) for l in LOWERCASE_LATIN]
    batch = torch.stack(images, dim=0).unsqueeze(-1).to(device)
    with torch.no_grad():
        _, embedding = model(batch)
    raw = embedding.cpu().numpy()
    unit = raw / (np.linalg.norm(raw, axis=-1, keepdims=True) + 1e-8)
    return unit


def main():
    rng = random.Random(7)
    byName = {}
    for directory, ext in SOURCES:
        if os.path.isdir(directory):
            byName.update(collectExistingPaths(directory, ext))
    complete = [n for n, letters in byName.items() if all(l in letters for l in LOWERCASE_LATIN)]
    fontNames = rng.sample(complete, 4)
    print(f"Sampled fonts: {fontNames}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = ViT.load(CHECKPOINT_DIR)
    model = model.to(device).eval()

    unitLetters = {name: letterEmbeddings(model, device, byName[name]) for name in fontNames}

    print(f"\n{'font':30s} |ubar| (current storage norm)")
    for name in fontNames:
        ubar = unitLetters[name].mean(axis=0)
        print(f"  {name[:30]:30s} {np.linalg.norm(ubar):.4f}")

    print(f"\n{'pair':45s} {'mean pairwise cos':>18s} {'cos(mean, mean)':>18s} {'delta':>8s}")
    for i in range(len(fontNames)):
        for j in range(i + 1, len(fontNames)):
            a, b = unitLetters[fontNames[i]], unitLetters[fontNames[j]]
            meanPairwiseCos = float((a @ b.T).mean())  # mean_{k,l} u_k . v_l

            ubar, vbar = a.mean(axis=0), b.mean(axis=0)
            cosOfMeans = float((ubar @ vbar) / (np.linalg.norm(ubar) * np.linalg.norm(vbar) + 1e-8))

            label = f"{fontNames[i][:20]} / {fontNames[j][:20]}"
            print(f"  {label:45s} {meanPairwiseCos:18.4f} {cosOfMeans:18.4f} {cosOfMeans - meanPairwiseCos:8.4f}")


if __name__ == "__main__":
    main()
