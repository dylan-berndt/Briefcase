"""
Three follow-up checks on the font-vs-Flickr8k density gap found by
geometry_comparison.py:

1. Does removing near-duplicate fonts fix the density gap, or is tightness
   pervasive across the whole corpus?
2. embeddings/all.json (used throughout this investigation) was generated
   directly off checkpoints/pretrain/best -- NO sigreg/contrastive step
   (verified: file timestamps land 1 second apart, and utils/training.py's
   sigreg_weak_loss only runs inside EmbeddingLoss, the finetune-stage
   contrastive loss, never touched during pretraining). embeddings/
   allText.json IS the actual sigreg-regularized contrastive output
   (timestamp matches checkpoints/finetune/2026-06-07 17-04 exactly). Does
   sigreg, where it was actually applied, produce meaningfully less-tight
   embeddings?
3. Does the LSA/TF-IDF text embedding space show the same tightness as BGE?

    python3 experiments/embedding-geometry/followup_diagnostics.py
"""
import json
import os
import pickle

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def nearestNeighborCosine(values, sampleSize=2000, seed=0, excludeTopFraction=0.0):
    rng = np.random.RandomState(seed)
    n = values.shape[0]
    normed = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)

    if excludeTopFraction > 0:
        # compute each point's OWN nearest-neighbor cosine (full corpus), then drop the
        # tightest excludeTopFraction of points (the "near-duplicates") before resampling
        chunk = 4000
        nnAll = np.empty(n, dtype=np.float32)
        for start in range(0, n, chunk):
            block = normed[start:start + chunk] @ normed.T
            for i in range(block.shape[0]):
                block[i, start + i] = -1
            nnAll[start:start + chunk] = block.max(axis=1)
        keepThreshold = np.quantile(nnAll, 1 - excludeTopFraction)
        keepIdx = np.where(nnAll <= keepThreshold)[0]
        normed = normed[keepIdx]
        n = normed.shape[0]

    sampleIdx = rng.choice(n, min(sampleSize, n), replace=False)
    sims = normed[sampleIdx] @ normed.T
    for i, idx in enumerate(sampleIdx):
        sims[i, idx] = -1
    nearest = sims.max(axis=1)
    return float(nearest.mean()), float(np.median(nearest)), n


def check1_nearDuplicates():
    print("=== 1. Effect of removing near-duplicate fonts ===")
    with open(os.path.join(REPO_ROOT, "embeddings", "all.json")) as f:
        fontVisual = json.load(f)
    values = np.array(list(fontVisual.values()), dtype=np.float64)

    for excludeFrac in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
        mean, median, n = nearestNeighborCosine(values, excludeTopFraction=excludeFrac)
        print(f"  exclude tightest {excludeFrac:.0%}: n={n}  NN-cosine mean={mean:.4f} median={median:.4f}")
    print("  (Flickr8k reference, for comparison: NN-cosine mean=0.6262 median=0.6167)")


def check2_sigreg():
    print("\n=== 2. all.json (pretrain, no sigreg) vs allText.json (finetuned, sigreg-regularized) ===")
    with open(os.path.join(REPO_ROOT, "embeddings", "all.json")) as f:
        allJson = json.load(f)
    with open(os.path.join(REPO_ROOT, "embeddings", "allText.json")) as f:
        allTextJson = json.load(f)

    commonNames = sorted(set(allJson.keys()) & set(allTextJson.keys()))
    print(f"  {len(commonNames)} fonts in common")

    allValues = np.array([allJson[n] for n in commonNames], dtype=np.float64)
    allTextValues = np.array([allTextJson[n] for n in commonNames], dtype=np.float64)

    mean1, median1, _ = nearestNeighborCosine(allValues)
    mean2, median2, _ = nearestNeighborCosine(allTextValues)
    print(f"  all.json      (pretrain, no sigreg):        NN-cosine mean={mean1:.4f} median={median1:.4f}")
    print(f"  allText.json  (finetuned, sigreg-regularized): NN-cosine mean={mean2:.4f} median={median2:.4f}")


def check3_lsa():
    print("\n=== 3. LSA/TF-IDF text embeddings vs BGE text embeddings ===")
    tfidfPath = os.path.join(REPO_ROOT, "embeddings", "tfidfFeatures_d256.pkl")
    if not os.path.exists(tfidfPath):
        print(f"  {tfidfPath} not found, skipping")
        return
    with open(tfidfPath, "rb") as f:
        tfidfCache = pickle.load(f)
    values = np.concatenate([np.asarray(v) for v in tfidfCache.values()], axis=0).astype(np.float64)
    mean, median, n = nearestNeighborCosine(values)
    print(f"  LSA (d=256): n={n} dim={values.shape[1]} NN-cosine mean={mean:.4f} median={median:.4f}")
    print("  (BGE font text, for comparison: NN-cosine mean=0.9375 median=0.9403)")
    print("  (Flickr8k BGE captions, for comparison: NN-cosine mean=0.8505 median=0.8505)")


if __name__ == "__main__":
    check1_nearDuplicates()
    check2_sigreg()
    check3_lsa()
