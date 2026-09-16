"""
How much of the visual ViT embedding's nominal 512 dimensions are actually
"used"? Reads the already-computed embeddings/font_vit_embeddings.json (no
extra ViT forward passes) and reports:

  - PCA cumulative explained variance thresholds (dims needed for 90/95/99%)
  - participation ratio: (sum(eigenvalues))^2 / sum(eigenvalues^2), a
    continuous "effective rank" that doesn't require picking a threshold

Read-only diagnostic for the later text-vs-visual modality comparison.

    python3 experiments/effective_dimension.py
"""
import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=os.path.join("embeddings", "font_vit_embeddings.json"))
    args = parser.parse_args()

    with open(args.path) as f:
        embeddings = json.load(f)

    values = np.array(list(embeddings.values()), dtype=np.float64)
    print(f"{values.shape[0]} fonts, dimension {values.shape[1]}")

    centered = values - values.mean(axis=0, keepdims=True)
    _, singularValues, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singularValues ** 2) / (values.shape[0] - 1)

    explainedVarianceRatio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explainedVarianceRatio)
    participationRatio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

    print(f"\nNominal dimension: {values.shape[1]}")
    for threshold in [0.90, 0.95, 0.99]:
        dims = int(np.searchsorted(cumulative, threshold) + 1)
        print(f"  dims to reach {threshold:.0%} cumulative variance: {dims}")
    print(f"  participation ratio (effective rank): {participationRatio:.1f}")
    print(f"  top-5 explained variance ratios: {explainedVarianceRatio[:5].round(4).tolist()}")


if __name__ == "__main__":
    main()
