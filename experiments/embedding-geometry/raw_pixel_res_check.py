"""
Same raw-pixel differentiation test (participation ratio, dims95, NN cosine) as
raw_pixel_differentiation.py / raw_pixel_google_only.py, but redirected to the 64px ("_64") bitmap
caches instead of the default 48px ones -- checks whether the very low measured differentiation
(PR~2-3) is a rendering-resolution artifact (fine stylistic detail lost to a coarse raster) or a
real property of the font designs themselves, independent of render size.

    python experiments/embedding-geometry/raw_pixel_res_check.py [--google-only]
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
SUFFIX = "_64"
GOOGLE_ONLY = "--google-only" in _sys.argv

with open("embeddings/fontGlyphPaths.pkl", "rb") as f:
    pathMap = pickle.load(f)


def source(n):
    p = list(pathMap[n].values())[0].replace("\\", "/")
    return "google" if p.startswith("google/") else ("dafont" if p.startswith("dafont/") else "myfonts")


def redirect(path):
    head, tail = _os.path.split(path)
    src, folder = _os.path.split(head)
    return _os.path.join(src, folder + SUFFIX, tail)


names = sorted(n for n in pathMap if all(l in pathMap[n] for l in LETTERS))
if GOOGLE_ONLY:
    names = [n for n in names if source(n) == "google"]
else:
    rng = np.random.RandomState(0)
    names = list(rng.choice(names, min(8000, len(names)), replace=False))

# verify redirected paths exist for the sample; drop any font missing a 64px render
usable = [n for n in names if all(_os.path.exists(redirect(pathMap[n][l])) for l in LETTERS[:3])]
print(f"{len(names)} candidate fonts ({'google-only' if GOOGLE_ONLY else 'full-corpus sample'}), {len(usable)} with 64px renders available", flush=True)
names = usable

vecs = []
for k, n in enumerate(names):
    imgs = [centerAndScaleNormalize(loadBitmap(redirect(pathMap[n][l])), outSize=64) for l in LETTERS]
    vecs.append(np.mean([im.flatten() for im in imgs], axis=0))
    if (k + 1) % 500 == 0:
        print(f"\r  {k + 1}/{len(names)}", end="", flush=True)
print()
vecs = np.array(vecs, dtype=np.float64)

pr, dims95 = participationRatio(vecs)
nnMean, nnMedian = nearestNeighborCosine(vecs, sampleSize=min(2000, len(vecs)))
label = "GOOGLE FONTS ONLY" if GOOGLE_ONLY else "FULL CORPUS SAMPLE"
print(f"\n{label}, 64px raw pixel (26-letter-averaged, {len(vecs)} fonts, {vecs.shape[1]}-dim):")
print(f"  participation ratio (effective dim) = {pr:.1f} / {vecs.shape[1]}   ({pr/vecs.shape[1]*100:.1f}% of ambient dim)")
print(f"  dims for 95% variance = {dims95}")
print(f"  nearest-neighbor cosine: mean={nnMean:.4f} median={nnMedian:.4f}")
