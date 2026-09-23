"""
Compares the SAME geometric diagnostics across two decoupled-encoder
corpora: Flickr8k (DINOv2 images + BGE captions, known-working reference)
and this project's own fonts (project ViT + BGE queries). Reuses the exact
participation-ratio method from experiments/diffusion-searches/diagnostics/
effective_dimension.py, plus a nearest-neighbor cosine density check
directly comparable to this project's own corpus-density finding (real
fonts sit at ~0.96 cosine to their nearest neighbor).

    python3 experiments/embedding-geometry/geometry_comparison.py
"""
import json
import os
import pickle

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

FLICKR_IMAGE_EMB = os.path.join(SCRIPT_DIR, "data", "image_embeddings.pkl")
FLICKR_TEXT_EMB = os.path.join(SCRIPT_DIR, "data", "text_embeddings.pkl")
FONT_EMBEDDINGS_PATH = os.path.join(REPO_ROOT, "embeddings", "all.json")
FONT_TEXT_EMB_PATH = os.path.join(REPO_ROOT, "embeddings", "sentenceQueries_BAAI_bge-large-en-v1.5.pkl")


def participationRatio(values):
    """Same method as experiments/diffusion-searches/diagnostics/effective_dimension.py."""
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    _, singularValues, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singularValues ** 2) / (values.shape[0] - 1)
    ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    explainedVarianceRatio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explainedVarianceRatio)
    dims95 = int(np.searchsorted(cumulative, 0.95) + 1)
    return ratio, dims95


def nearestNeighborCosine(values, sampleSize=2000, seed=0):
    """Mean/median cosine similarity of each sampled point to its single nearest OTHER point."""
    rng = np.random.RandomState(seed)
    n = values.shape[0]
    sampleIdx = rng.choice(n, min(sampleSize, n), replace=False)
    normed = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)
    sims = normed[sampleIdx] @ normed.T
    for i, idx in enumerate(sampleIdx):
        sims[i, idx] = -1  # exclude self
    nearest = sims.max(axis=1)
    return float(nearest.mean()), float(np.median(nearest))


def report(label, imageOrVisual, text):
    visRatio, visDims95 = participationRatio(imageOrVisual)
    txtRatio, txtDims95 = participationRatio(text)
    visMean, visMedian = nearestNeighborCosine(imageOrVisual)
    txtMean, txtMedian = nearestNeighborCosine(text)

    print(f"\n=== {label} ===")
    print(f"  visual/image: n={imageOrVisual.shape[0]} dim={imageOrVisual.shape[1]} "
          f"participation_ratio={visRatio:.1f} (95%% var needs {visDims95} dims) "
          f"NN-cosine mean={visMean:.4f} median={visMedian:.4f}")
    print(f"  text:         n={text.shape[0]} dim={text.shape[1]} "
          f"participation_ratio={txtRatio:.1f} (95%% var needs {txtDims95} dims) "
          f"NN-cosine mean={txtMean:.4f} median={txtMedian:.4f}")


def main():
    print("loading Flickr8k embeddings...")
    with open(FLICKR_IMAGE_EMB, "rb") as f:
        flickrImages = pickle.load(f)
    with open(FLICKR_TEXT_EMB, "rb") as f:
        flickrTexts = pickle.load(f)

    imageMatrix = np.stack(list(flickrImages.values()))
    textMatrix = np.concatenate([np.asarray(v) for v in flickrTexts.values()], axis=0)
    report("FLICKR8K (DINOv2 images / BGE captions)", imageMatrix, textMatrix)

    print("\nloading font embeddings...")
    with open(FONT_EMBEDDINGS_PATH) as f:
        fontVisual = json.load(f)
    with open(FONT_TEXT_EMB_PATH, "rb") as f:
        fontText = pickle.load(f)

    visualMatrix = np.array(list(fontVisual.values()), dtype=np.float64)
    fontTextMatrix = np.concatenate([np.asarray(v) for v in fontText.values()], axis=0)
    report("FONTS (project ViT visual / BGE queries)", visualMatrix, fontTextMatrix)


if __name__ == "__main__":
    main()
