"""
Same raw-pixel differentiation test as raw_pixel_differentiation.py, restricted to Google Fonts only
(a smaller, curated set) -- to check whether the low measured differentiation (PR=3.3/2304) is a
property of the WHOLE mixed corpus (google+dafont+myfonts) or specifically diluted by the much larger,
less-curated dafont/myfonts sources.

    python experiments/embedding-geometry/raw_pixel_google_only.py
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
from ssim_duplicates import loadBitmap, centerAndScaleNormalize
from raw_pixel_differentiation import participationRatio, nearestNeighborCosine

LETTERS = list("abcdefghijklmnopqrstuvwxyz")

with open("embeddings/fontGlyphPaths.pkl", "rb") as f:
    pathMap = pickle.load(f)


def source(n):
    p = list(pathMap[n].values())[0].replace("\\", "/")
    return "google" if p.startswith("google/") else ("dafont" if p.startswith("dafont/") else "myfonts")


names = sorted(n for n in pathMap if all(l in pathMap[n] for l in LETTERS) and source(n) == "google")
print(f"{len(names)} Google Fonts with all 26 letters", flush=True)

vecs = []
for k, n in enumerate(names):
    imgs = [centerAndScaleNormalize(loadBitmap(pathMap[n][l])) for l in LETTERS]
    vecs.append(np.mean([im.flatten() for im in imgs], axis=0))
    if (k + 1) % 500 == 0:
        print(f"\r  {k + 1}/{len(names)}", end="", flush=True)
print()
vecs = np.array(vecs, dtype=np.float64)

pr, dims95 = participationRatio(vecs)
nnMean, nnMedian = nearestNeighborCosine(vecs, sampleSize=min(2000, len(vecs)))
print(f"\nGOOGLE FONTS ONLY, raw pixel (26-letter-averaged, {len(vecs)} fonts, {vecs.shape[1]}-dim):")
print(f"  participation ratio (effective dim) = {pr:.1f} / {vecs.shape[1]}   ({pr/vecs.shape[1]*100:.1f}% of ambient dim)")
print(f"  dims for 95% variance = {dims95}")
print(f"  nearest-neighbor cosine: mean={nnMean:.4f} median={nnMedian:.4f}")
print(f"\n  for comparison, FULL mixed corpus (8000-font sample, same method): PR=3.3  dims95=39  NNcos mean=0.972")
