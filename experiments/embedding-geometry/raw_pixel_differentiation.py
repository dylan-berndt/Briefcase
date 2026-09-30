"""
Reliability (split_half_reliability.py, r=0.92) shows font-pair similarity is CONSISTENT across
different letters -- but that's compatible with the corpus actually being just a couple of true
style clusters plus noise (consistency doesn't imply richness: if there were only 2 real styles in
40k fonts, split-half reliability would still be ~1.0). This measures the other half of the question:
how much real DIFFERENTIATION exists, using the same participation-ratio (effective dimension) and
nearest-neighbor-cosine (corpus density) metrics already used throughout this investigation for
trained embeddings (all.json: PR=57.6, NN cosine=0.962) -- but here computed on RAW PIXELS, no model,
no training, before anything could have collapsed or spread the space.

    python experiments/embedding-geometry/raw_pixel_differentiation.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)
_os.chdir(_repoRoot)
_sys.path.insert(0, _scriptDir)

import pickle

import numpy as np
import torch

from ssim_duplicates import loadBitmap, centerAndScaleNormalize

LETTERS = list("abcdefghijklmnopqrstuvwxyz")
N_FONTS = 8000   # matched in spirit to the 8091-font ASIF scale used throughout this session


def participationRatio(values):
    centered = values - values.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    eig = (s ** 2) / (values.shape[0] - 1)
    ratio = (eig.sum() ** 2) / (eig ** 2).sum()
    cum = np.cumsum(eig / eig.sum())
    dims95 = int(np.searchsorted(cum, 0.95) + 1)
    return ratio, dims95


def nearestNeighborCosine(values, sampleSize=2000, seed=0):
    rng = np.random.RandomState(seed)
    n = values.shape[0]
    idx = rng.choice(n, min(sampleSize, n), replace=False)
    normed = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)
    sims = normed[idx] @ normed.T
    for i, j in enumerate(idx):
        sims[i, j] = -1
    nearest = sims.max(axis=1)
    return float(nearest.mean()), float(np.median(nearest))


def main():
    with open("embeddings/fontGlyphPaths.pkl", "rb") as f:
        pathMap = pickle.load(f)
    names = sorted(n for n in pathMap if all(l in pathMap[n] for l in LETTERS))
    rng = np.random.RandomState(0)
    sampleNames = rng.choice(names, min(N_FONTS, len(names)), replace=False)
    print(f"{len(names)} fonts with all 26 letters, sampling {len(sampleNames)}", flush=True)

    vecs = []
    for k, n in enumerate(sampleNames):
        imgs = [centerAndScaleNormalize(loadBitmap(pathMap[n][l])) for l in LETTERS]
        vecs.append(np.mean([im.flatten() for im in imgs], axis=0))
        if (k + 1) % 1000 == 0:
            print(f"\r  {k + 1}/{len(sampleNames)}", end="", flush=True)
    print()
    vecs = np.array(vecs, dtype=np.float64)

    pr, dims95 = participationRatio(vecs)
    nnMean, nnMedian = nearestNeighborCosine(vecs)
    print(f"\nRAW PIXEL (26-letter-averaged, centered/scaled per glyph, {len(vecs)} fonts, {vecs.shape[1]}-dim):")
    print(f"  participation ratio (effective dim) = {pr:.1f} / {vecs.shape[1]}   ({pr/vecs.shape[1]*100:.1f}% of ambient dim)")
    print(f"  dims for 95% variance = {dims95}")
    print(f"  nearest-neighbor cosine: mean={nnMean:.4f} median={nnMedian:.4f}")
    print(f"\n  for comparison, TRAINED embeddings measured this session on comparable/full corpus:")
    print(f"    all.json (best, 512-dim):        PR=57.6  dims95=97   NN cosine mean=0.962")
    print(f"    weak-sigreg backbone (256-dim):   PR=33.0  dims95=54   NN cosine mean=0.952")
    print(f"    deployed allText.json (512-dim):  PR=8.0   dims95=13   NN cosine mean=0.988")


if __name__ == "__main__":
    main()
