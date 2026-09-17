"""
Is mean-pooling over all 26 lowercase letters washing out real signal?
generateEmbeddings averages each font's 26 per-letter CLS embeddings
(after per-letter L2 normalization) into one vector, and that pooled
space turned out to have an effective rank of ~9-10 out of 512 across
~40k fonts. This checks whether that's a property of the *pooling*
or of the *underlying per-letter embeddings themselves*, by comparing,
on the same small sample of fonts:

  (a) effective dim of all (font, letter) CLS embeddings pooled together
      -- i.e. before any per-font averaging (includes letter-identity variance)
  (b) effective dim of a SINGLE fixed letter's embedding across fonts
      -- isolates font-style variance for one letter, pre-pooling
  (c) effective dim of the actual per-font mean-pooled embedding
      -- reproduces generateEmbeddings exactly, for direct comparison
  (d) a variance decomposition: how much of total per-letter embedding
      variance is between-font vs. within-font(letter-to-letter)

If (b) has meaningfully higher effective rank than (c), mean-pooling
across letters is destroying font-style variance that individual
letters do carry.

    python3 experiments/character_variance.py --sample 300
"""
import argparse
import os
import random
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from utils.pretraining import characters, loadImage
from utils.vit import ViT

CHECKPOINT_DIR = os.path.join("checkpoints", "pretrain", "best")
LOWERCASE_LATIN = [chr(c) for c in range(ord('a'), ord('z') + 1)]

SOURCES = [
    ("google", os.path.join("google", "bitmaps"), ".bmp"),
    ("dafont", os.path.join("dafont", "bitmaps"), ".bmp"),
    ("myfonts", os.path.join("dataset", "smallimage"), ".bmp"),
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
        name = stem[:-3]
        byName.setdefault(name, {})[char] = path
    return byName


def effectiveDim(values):
    centered = values - values.mean(axis=0, keepdims=True)
    _, singularValues, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singularValues ** 2) / max(values.shape[0] - 1, 1)
    explainedVarianceRatio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explainedVarianceRatio)
    participationRatio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    dims95 = int(np.searchsorted(cumulative, 0.95) + 1)
    return participationRatio, dims95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    byName = {}
    for label, directory, ext in SOURCES:
        if not os.path.isdir(directory):
            continue
        byName.update(collectExistingPaths(directory, ext))

    complete = [name for name, letters in byName.items() if all(l in letters for l in LOWERCASE_LATIN)]
    sampleNames = rng.sample(complete, min(args.sample, len(complete)))
    print(f"Sampling {len(sampleNames)} fonts (of {len(complete)} with a complete a-z set)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = ViT.load(CHECKPOINT_DIR)
    model = model.to(device)
    model.eval()

    # raw[i, j] = CLS embedding for sampleNames[i], LOWERCASE_LATIN[j] -- no
    # per-letter normalization applied yet, unlike generateEmbeddings.
    raw = np.zeros((len(sampleNames), len(LOWERCASE_LATIN), 512), dtype=np.float32)
    with torch.no_grad():
        for i, name in enumerate(sampleNames):
            images = [torch.tensor(loadImage(byName[name][letter])[1], dtype=torch.float32)
                      for letter in LOWERCASE_LATIN]
            batch = torch.stack(images, dim=0).unsqueeze(-1).to(device)
            _, embedding = model(batch)
            raw[i] = embedding.cpu().numpy()
            print(f"\r{i + 1}/{len(sampleNames)} fonts embedded", end="")
    print()

    normalized = raw / (np.linalg.norm(raw, axis=-1, keepdims=True) + 1e-8)

    # (a) every (font, letter) pair as its own point
    allPairs = normalized.reshape(-1, 512)
    prA, d95A = effectiveDim(allPairs)

    # (b) a handful of single fixed letters, across all sampled fonts
    fixedLetterResults = {}
    for letter in ["a", "m", "z"]:
        idx = LOWERCASE_LATIN.index(letter)
        pr, d95 = effectiveDim(normalized[:, idx, :])
        fixedLetterResults[letter] = (pr, d95)

    # (c) reproduce generateEmbeddings: normalize per-letter, mean over letters
    pooled = normalized.mean(axis=1)
    prC, d95C = effectiveDim(pooled)

    # (d) variance decomposition in the per-letter-normalized space
    grandMean = normalized.reshape(-1, 512).mean(axis=0)
    betweenFont = ((pooled - grandMean) ** 2).sum(axis=1).mean() * len(LOWERCASE_LATIN)
    withinFont = ((normalized - pooled[:, None, :]) ** 2).sum(axis=-1).mean(axis=0).sum()
    totalVariance = betweenFont + withinFont

    print(f"\n(a) all (font, letter) pairs pooled together: "
          f"participation ratio {prA:.1f}, dims for 95% variance {d95A}")
    print("(b) single fixed letter, across fonts (isolates font-style variance pre-pooling):")
    for letter, (pr, d95) in fixedLetterResults.items():
        print(f"      letter '{letter}': participation ratio {pr:.1f}, dims for 95% variance {d95}")
    print(f"(c) per-font mean-pooled embedding (reproduces generateEmbeddings): "
          f"participation ratio {prC:.1f}, dims for 95% variance {d95C}")
    print(f"\n(d) variance decomposition (per-letter-normalized space):")
    print(f"      between-font variance: {betweenFont:.4f}")
    print(f"      within-font (letter-to-letter) variance: {withinFont:.4f}")
    print(f"      fraction of total variance that is between-font: {betweenFont / totalVariance:.4f}")


if __name__ == "__main__":
    main()
