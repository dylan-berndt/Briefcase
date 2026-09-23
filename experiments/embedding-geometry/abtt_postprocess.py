"""
All-but-the-Top (Mu & Viswanath, 2018, "All-but-the-Top: Simple and
Effective Postprocessing for Word Representations"): a training-free fix
for anisotropic embeddings. Center the embeddings, then remove each
vector's projection onto the top-D dominant principal directions (the
directions carrying the most variance, which for anisotropic spaces tend
to be shared/generic rather than discriminative), leaving the remaining,
more genuinely-discriminative structure. No retraining, just linear
algebra on the existing embeddings/all.json.

    python3 experiments/embedding-geometry/abtt_postprocess.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)

import json

import numpy as np


def participationRatio(values):
    centered = values - values.mean(axis=0, keepdims=True)
    _, singularValues, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singularValues ** 2) / (values.shape[0] - 1)
    return (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()


def nearestNeighborCosine(values, sampleSize=2000, seed=0):
    rng = np.random.RandomState(seed)
    n = values.shape[0]
    sampleIdx = rng.choice(n, min(sampleSize, n), replace=False)
    normed = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)
    sims = normed[sampleIdx] @ normed.T
    for i, idx in enumerate(sampleIdx):
        sims[i, idx] = -1
    nearest = sims.max(axis=1)
    return float(nearest.mean()), float(np.median(nearest))


def allButTheTop(values, numRemove):
    """values: [N, D]. Removes each vector's projection onto the top-`numRemove`
    principal directions (by variance) after centering. Returns the (still
    centered) result -- NOT re-normalized, matching the original paper."""
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    topDirections = vt[:numRemove]  # [numRemove, D]
    projections = centered @ topDirections.T  # [N, numRemove]
    removed = centered - projections @ topDirections
    return removed


def main():
    with open(_os.path.join(_repoRoot, "embeddings", "all.json")) as f:
        fontVisual = json.load(f)
    names = sorted(fontVisual.keys())
    values = np.array([fontVisual[n] for n in names], dtype=np.float64)

    baseRatio = participationRatio(values)
    baseMean, baseMedian = nearestNeighborCosine(values)
    print(f"baseline (all.json): participation_ratio={baseRatio:.1f} NN-cosine mean={baseMean:.4f} median={baseMedian:.4f}")

    for numRemove in [1, 2, 3, 5, 10, 20, 50]:
        processed = allButTheTop(values, numRemove)
        ratio = participationRatio(processed)
        mean, median = nearestNeighborCosine(processed)
        print(f"  remove top-{numRemove}: participation_ratio={ratio:.1f} NN-cosine mean={mean:.4f} median={median:.4f}")

    print("\n(Flickr8k reference: NN-cosine mean=0.6262 median=0.6167)")


if __name__ == "__main__":
    main()
