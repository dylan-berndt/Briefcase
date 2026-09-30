"""
Model-independent near-duplicate check: does raw glyph-image similarity (SSIM, after a per-pair
black-box search over scale+offset to align the two images) agree with what the trained embeddings
say? Motivation: the trained embeddings' own loss (InfoNCE) is shown (this session) to apply its
LARGEST separating gradient specifically to whichever negative currently looks most similar -- so an
embedding-based near-duplicate check can systematically miss true near-duplicates the training
actively pushed apart. SSIM on raw pixels never saw that loss, so it can't be fooled the same way.

Two stages, since brute-force alignment+SSIM over all ~39,421 choose 2 pairs is infeasible:
  1. Cheap, ALSO model-independent prefilter: raw-pixel cosine similarity (flattened bitmap, glyph
     'a', 48px) via the same chunked-GPU-matmul approach as prune_duplicates.py -- narrows ~777M
     possible pairs down to a candidate list.
  2. For each candidate pair: black-box optimize (scipy.optimize, Powell, no gradients) over
     (scale, dx, dy) to align one glyph to the other, maximizing SSIM -- searched over both 'a' and
     'g' where available, averaged. Search ranges are narrow (per the user: fonts are "mostly well
     centered", this only needs to catch accidental small scale/offset differences), so this is fast.

Reports the aligned-SSIM pruning table (same format as prune_duplicates.py) restricted to this
candidate set, AND, more importantly, the trained embeddings' OWN cosine similarity for pairs found
to be true near-duplicates by SSIM -- directly showing how many were hidden by training.

    python experiments/embedding-geometry/ssim_duplicates.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)
_os.chdir(_repoRoot)

import csv
import json
import pickle
from multiprocessing import Pool

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.ndimage import affine_transform
from skimage.metrics import structural_similarity as ssim

CHUNK = 4000
PREFILTER_GLYPH = "a"
GLYPHS = ["a", "g"]
PREFILTER_THRESHOLD = float(_os.environ.get("PREFILTER_THRESHOLD", "0.99"))   # raw-pixel cosine, glyph 'a'
CANDIDATES_CACHE = _os.path.join("experiments", "embedding-geometry", "ssim_candidates.pkl")
RESULTS_CSV = _os.path.join("experiments", "embedding-geometry", "ssim_duplicate_pairs.csv")


def loadBitmap(path):
    img = np.fromfile(path, dtype=np.uint8)
    import cv2
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float64) / 255.0 if img is not None else None


def centerAndScaleNormalize(img, outSize=48):
    """Per-IMAGE normalization (no pairing involved): recenter on the image's own ink centroid and
    rescale so its own ink bounding box fills a fixed fraction of the canvas. This is what makes the
    cheap prefilter shift/scale-INVARIANT -- two accidentally-shifted/scaled copies of the same glyph
    map to nearly the same normalized image, without ever comparing them to each other."""
    mass = img.sum()
    if mass < 1e-6:
        return img
    ys, xs = np.nonzero(img > 0.05)
    if len(xs) == 0:
        return img
    cy, cx = ys.mean(), xs.mean()
    h = max(ys.max() - ys.min(), 1)
    w = max(xs.max() - xs.min(), 1)
    scale = (outSize * 0.6) / max(h, w)   # target ink extent ~60% of canvas, same for every glyph
    matrix = np.eye(2) / scale
    center = np.array([outSize / 2, outSize / 2])
    offset = np.array([cy, cx]) - matrix @ center
    return affine_transform(img, matrix, offset=offset, output_shape=(outSize, outSize), order=1, mode="constant", cval=0.0)


def collectPrefilterEdges(vecs, device):
    n = vecs.shape[0]
    v = torch.nn.functional.normalize(torch.tensor(vecs, dtype=torch.float32, device=device), dim=-1)
    allI, allJ, allS = [], [], []
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        block = v[start:end] @ v.T
        rowIdx = torch.arange(start, end, device=device).unsqueeze(1)
        colIdx = torch.arange(n, device=device).unsqueeze(0)
        mask = (block > PREFILTER_THRESHOLD) & (colIdx > rowIdx)
        ii, jj = torch.nonzero(mask, as_tuple=True)
        if len(ii):
            allI.append((ii + start).cpu()); allJ.append(jj.cpu()); allS.append(block[ii, jj].cpu())
        print(f"\r  prefilter rows {end}/{n}, candidates so far: {sum(len(x) for x in allI)}", end="", flush=True)
    print()
    if not allI:
        return np.array([], dtype=int), np.array([], dtype=int)
    return torch.cat(allI).numpy(), torch.cat(allJ).numpy()


def alignedSsim(imgA, imgB):
    """Black-box (Powell, derivative-free) search over (scale, dx, dy) transforming imgB onto imgA,
    maximizing SSIM. Narrow bounds -- images are pre-centered, this only needs to catch small
    accidental scale/offset differences."""
    h, w = imgA.shape
    center = np.array([h / 2, w / 2])

    def negSsim(params):
        scale, dx, dy = params
        matrix = np.eye(2) / scale
        offset = center - matrix @ center - np.array([dy, dx])
        warped = affine_transform(imgB, matrix, offset=offset, output_shape=imgA.shape,
                                   order=1, mode="constant", cval=0.0)
        return -ssim(imgA, warped, data_range=1.0)

    result = minimize(negSsim, x0=[1.0, 0.0, 0.0], method="Powell",
                       bounds=[(0.85, 1.15), (-6, 6), (-6, 6)],
                       options={"xtol": 0.01, "ftol": 0.001, "maxiter": 60})
    return -result.fun


def evalPair(args):
    pathsA, pathsB = args
    scores = []
    for g in GLYPHS:
        if g not in pathsA or g not in pathsB:
            continue
        a, b = loadBitmap(pathsA[g]), loadBitmap(pathsB[g])
        if a is None or b is None or a.shape != b.shape:
            continue
        scores.append(alignedSsim(a, b))
    return float(np.mean(scores)) if scores else None


def cosine(embeds, name, a, b):
    if a not in embeds[name] or b not in embeds[name]:
        return None
    va, vb = np.array(embeds[name][a]), np.array(embeds[name][b])
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open("embeddings/fontGlyphPaths.pkl", "rb") as f:
        pathMap = pickle.load(f)

    # Stage 1 (prefilter): cached to disk immediately, so a crash/interrupt in stage 2 never means
    # re-scanning ~39.7k bitmaps again.
    if _os.path.exists(CANDIDATES_CACHE):
        print(f"loading cached candidates from {CANDIDATES_CACHE}", flush=True)
        with open(CANDIDATES_CACHE, "rb") as f:
            names, edgeI, edgeJ, prefilterCos = pickle.load(f)
    else:
        names = sorted(n for n in pathMap if all(g in pathMap[n] for g in GLYPHS))
        print(f"{len(names)} fonts with both {GLYPHS}", flush=True)

        print("loading + centroid/scale-normalizing glyph 'a' bitmaps for prefilter (shift/scale-invariant)...", flush=True)
        vecs = []
        for n in names:
            img = loadBitmap(pathMap[n][PREFILTER_GLYPH])
            vecs.append(centerAndScaleNormalize(img).flatten() if img is not None else np.zeros(48 * 48))
        vecs = np.stack(vecs)
        print(f"prefilter vectors: {vecs.shape}", flush=True)

        edgeI, edgeJ = collectPrefilterEdges(vecs, device)
        print(f"{len(edgeI)} candidate pairs above prefilter threshold {PREFILTER_THRESHOLD} (raw pixel cosine, glyph 'a')", flush=True)
        if len(edgeI) > 200_000:
            print(f"too many candidates ({len(edgeI)}) for the alignment stage -- raise PREFILTER_THRESHOLD and rerun", flush=True)
            return
        v = torch.nn.functional.normalize(torch.tensor(vecs, dtype=torch.float32), dim=-1)
        prefilterCos = (v[edgeI] * v[edgeJ]).sum(dim=1).numpy() if len(edgeI) else np.array([])
        with open(CANDIDATES_CACHE, "wb") as f:
            pickle.dump((names, edgeI, edgeJ, prefilterCos), f)
        print(f"cached candidates to {CANDIDATES_CACHE}", flush=True)

    if len(edgeI) == 0:
        print("no candidates.")
        return

    # trained-embedding cosine for the same pairs -- included as columns, not used to filter anything
    embeds = {}
    for embName in ["all", "allText", "all_weak_sigreg_step85500"]:
        path = f"embeddings/{embName}.json"
        if _os.path.exists(path):
            with open(path) as f:
                embeds[embName] = json.load(f)

    # Stage 2 (expensive): black-box aligned SSIM per candidate pair, written to CSV as each result
    # arrives -- an interrupt loses at most the in-flight chunk, not the whole run. Resumable: pairs
    # already present in RESULTS_CSV (by name pair) are skipped.
    done = set()
    writeHeader = not _os.path.exists(RESULTS_CSV)
    if not writeHeader:
        with open(RESULTS_CSV, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # header
            for row in reader:
                done.add((row[0], row[1]))
        print(f"{len(done)} pairs already in {RESULTS_CSV}, resuming", flush=True)

    pending = [(i, j) for i, j in zip(edgeI, edgeJ) if (names[i], names[j]) not in done]
    print(f"running black-box aligned SSIM on {len(pending)} pairs (of {len(edgeI)} total candidates)...", flush=True)
    tasks = [(pathMap[names[i]], pathMap[names[j]]) for i, j in pending]

    embCols = list(embeds.keys())
    with open(RESULTS_CSV, "a", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        if writeHeader:
            writer.writerow(["fontA", "fontB", "prefilterCos", "alignedSsim"] + [f"cos_{k}" for k in embCols])
        cosLookup = {(i, j): float(prefilterCos[k]) for k, (i, j) in enumerate(zip(edgeI, edgeJ))}
        with Pool(8) as pool:
            for k, r in enumerate(pool.imap(evalPair, tasks, chunksize=20)):
                i, j = pending[k]
                a, b = names[i], names[j]
                ssimVal = "" if r is None else f"{r:.6f}"
                cosVals = [cosine(embeds, name, a, b) for name in embCols]
                cosVals = ["" if c is None else f"{c:.6f}" for c in cosVals]
                writer.writerow([a, b, f"{cosLookup[(i, j)]:.6f}", ssimVal] + cosVals)
                if (k + 1) % 500 == 0:
                    out.flush()
                    print(f"\r  {k + 1}/{len(tasks)}", end="", flush=True)
    print(f"\ndone. full per-pair results in {RESULTS_CSV} -- load it (pandas/Excel/whatever) and pick your own threshold on the alignedSsim column.")


if __name__ == "__main__":
    main()
