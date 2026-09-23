"""
Tests whether mean-pooling across 26 letters is itself washing out real
per-font signal (distinct from the already-documented normalize-order bug,
which cosine similarity is invariant to and so can't explain the measured
NN-cosine tightness). Extracts UNPOOLED, single-letter embeddings directly
from checkpoints/pretrain/best's ViT for a sample of fonts and measures
their own NN-cosine density and participation ratio, then does the same
after averaging 2, 5, and all 26 letters -- if tightness increases
monotonically as more letters get pooled together, that directly implicates
averaging (not just the checkpoint/training) as a real contributor.

    python3 experiments/embedding-geometry/test_pooling_hypothesis.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)

import pickle

import numpy as np
import torch

from utils.vit import ViT
from utils.loaders.standard import loadImage

CHECKPOINT_DIR = _os.path.join(_repoRoot, "checkpoints", "pretrain", "best")
GLYPH_CACHE = _os.path.join(_repoRoot, "embeddings", "fontGlyphPaths.pkl")
LATIN = [chr(c) for c in range(ord('a'), ord('z') + 1)]


def participationRatio(values):
    centered = values - values.mean(axis=0, keepdims=True)
    _, singularValues, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singularValues ** 2) / (values.shape[0] - 1)
    return (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()


def nearestNeighborCosine(values, sampleSize=1500, seed=0):
    rng = np.random.RandomState(seed)
    n = values.shape[0]
    sampleIdx = rng.choice(n, min(sampleSize, n), replace=False)
    normed = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)
    sims = normed[sampleIdx] @ normed.T
    for i, idx in enumerate(sampleIdx):
        sims[i, idx] = -1
    nearest = sims.max(axis=1)
    return float(nearest.mean()), float(np.median(nearest))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = ViT.load(CHECKPOINT_DIR)
    model.to(device).eval()
    print(f"loaded {CHECKPOINT_DIR}")

    with open(GLYPH_CACHE, "rb") as f:
        pathMap = pickle.load(f)

    rng = np.random.RandomState(0)
    allNames = [n for n in pathMap if all(l in pathMap[n] for l in LATIN)]
    sampleNames = rng.choice(allNames, min(3000, len(allNames)), replace=False)
    print(f"{len(sampleNames)} sampled fonts (of {len(allNames)} with all 26 lowercase letters)")

    # perLetterEmbeddings[fontIdx] = [26, embedDim] raw CLS embeddings, UNNORMALIZED, UNPOOLED
    perLetterEmbeddings = np.zeros((len(sampleNames), 26, config.model.embedDim), dtype=np.float32)

    batchImages, batchIdx = [], []

    def flush():
        if not batchImages:
            return
        x = torch.stack(batchImages, dim=0).unsqueeze(-1).to(device)
        with torch.no_grad():
            _, c = model(x)
        c = c.cpu().numpy()
        for (fi, li), vec in zip(batchIdx, c):
            perLetterEmbeddings[fi, li] = vec
        batchImages.clear()
        batchIdx.clear()

    for fi, name in enumerate(sampleNames):
        for li, letter in enumerate(LATIN):
            path = pathMap[name][letter]
            _, img = loadImage(path)
            if img is None:
                continue
            batchImages.append(torch.tensor(img, dtype=torch.float32))
            batchIdx.append((fi, li))
            if len(batchImages) >= 512:
                flush()
        if (fi + 1) % 500 == 0:
            print(f"{fi + 1}/{len(sampleNames)}")
    flush()

    print("\n=== single, UNPOOLED letter (letter 'a' only) ===")
    single = perLetterEmbeddings[:, 0, :]
    ratio = participationRatio(single)
    mean, median = nearestNeighborCosine(single)
    print(f"  participation_ratio={ratio:.1f} NN-cosine mean={mean:.4f} median={median:.4f}")

    for numLetters in [2, 5, 10, 26]:
        pooled = nn_normalize_then_mean(perLetterEmbeddings[:, :numLetters, :])
        ratio = participationRatio(pooled)
        mean, median = nearestNeighborCosine(pooled)
        print(f"\n=== pooled over {numLetters} letters (normalize-then-mean, matches all.json's convention) ===")
        print(f"  participation_ratio={ratio:.1f} NN-cosine mean={mean:.4f} median={median:.4f}")

    print("\n(all.json full-corpus reference: participation_ratio=57.6 NN-cosine mean=0.9621)")
    print("(Flickr8k reference: NN-cosine mean=0.6262)")


def nn_normalize_then_mean(letterEmbeddings):
    """letterEmbeddings: [N, numLetters, D]. Matches utils/embeddings.py:51's exact convention:
    normalize each letter's embedding, then mean-pool (no final renormalize)."""
    norms = np.linalg.norm(letterEmbeddings, axis=-1, keepdims=True)
    normalized = letterEmbeddings / (norms + 1e-12)
    return normalized.mean(axis=1)


if __name__ == "__main__":
    main()
