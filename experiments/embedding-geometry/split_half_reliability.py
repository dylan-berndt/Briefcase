"""
Is there a real, letter-independent "style similarity" signal in the raw visual data at all, or is
apparent font-pair similarity mostly an accident of whichever single letter you happen to compare?
Split-half reliability test (standard measurement-theory technique): split the 26 lowercase letters
into two disjoint sets, compute each font's average normalized glyph vector separately within each
set, then correlate pairwise similarity computed from set A against pairwise similarity computed from
set B, over many random font pairs spanning the full similarity range (not just near-duplicates).

High correlation = a real, consistent, letter-independent property of the font pair (genuine style
signal a model could learn to respect). Low/no correlation = "similarity" is mostly per-letter noise,
which would explain why a continuous-similarity target (Soft-InfoNCE, RNC) struggles here the same way
it did with noisy single-query text targets earlier this session.

Model-independent: raw pixels only, no ViT, same centerAndScaleNormalize preprocessing already
validated in ssim_duplicates.py.

    python experiments/embedding-geometry/split_half_reliability.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)
_os.chdir(_repoRoot)
_sys.path.insert(0, _scriptDir)

import json
import pickle

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from ssim_duplicates import loadBitmap, centerAndScaleNormalize

SET_A = list("acegikmoqsuwy")   # 13 letters, alternating
SET_B = list("bdfhjlnprtvxz")   # the other 13, disjoint from SET_A
N_FONTS = 5000
N_PAIRS = 40000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open("embeddings/fontGlyphPaths.pkl", "rb") as f:
        pathMap = pickle.load(f)
    names = sorted(n for n in pathMap if all(l in pathMap[n] for l in SET_A + SET_B))
    print(f"{len(names)} fonts with all 26 lowercase letters", flush=True)

    rng = np.random.RandomState(0)
    sampleNames = rng.choice(names, min(N_FONTS, len(names)), replace=False)

    vecA, vecB = [], []
    for k, n in enumerate(sampleNames):
        imgsA = [centerAndScaleNormalize(loadBitmap(pathMap[n][l])) for l in SET_A]
        imgsB = [centerAndScaleNormalize(loadBitmap(pathMap[n][l])) for l in SET_B]
        vecA.append(np.mean([im.flatten() for im in imgsA], axis=0))
        vecB.append(np.mean([im.flatten() for im in imgsB], axis=0))
        if (k + 1) % 500 == 0:
            print(f"\r  {k + 1}/{len(sampleNames)} fonts loaded", end="", flush=True)
    print()
    vecA = np.stack(vecA); vecB = np.stack(vecB)

    def normalize(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

    vA = torch.tensor(normalize(vecA), dtype=torch.float32, device=device)
    vB = torch.tensor(normalize(vecB), dtype=torch.float32, device=device)

    n = len(sampleNames)
    i = rng.randint(0, n, N_PAIRS); j = rng.randint(0, n, N_PAIRS)
    mask = i != j; i, j = i[mask], j[mask]
    ii = torch.tensor(i, device=device); jj = torch.tensor(j, device=device)
    simA = (vA[ii] * vA[jj]).sum(dim=1).cpu().numpy()
    simB = (vB[ii] * vB[jj]).sum(dim=1).cpu().numpy()

    pear = pearsonr(simA, simB)
    spear = spearmanr(simA, simB)
    print(f"\nSplit-half reliability (letters {''.join(SET_A)} vs {''.join(SET_B)}), n_pairs={len(i)}")
    print(f"  Pearson  r = {pear.statistic:+.4f}  (p={pear.pvalue:.2e})")
    print(f"  Spearman r = {spear.statistic:+.4f}  (p={spear.pvalue:.2e})")
    print(f"  simA[mean,std] = [{simA.mean():.3f},{simA.std():.3f}]   simB[mean,std] = [{simB.mean():.3f},{simB.std():.3f}]")

    # sanity anchor: self-consistency (a font vs itself, trivially r=1) and known reference points
    print(f"\n  (for context: this session's earlier visual<->text correlation on the SAME kind of")
    print(f"   random-pair sample was Pearson +0.134 to +0.157 for fonts, +0.436 for Flickr8k)")


if __name__ == "__main__":
    main()
