"""
Near-duplicate analysis over one or more build_cache.py caches: how much of the corpus a similarity threshold t
would discard.

Similarity between two fonts = cosine similarity of their concatenated dedup glyphs (DEDUP_LETTERS, each
centerAndScaleNormalize'd, 8 x 48 x 48 = 18,432 pixels). Fonts missing any dedup letter can't be compared, and fonts
that draw the same placeholder box for every letter (see interLetterSimilarity) aren't fonts; both are left out of
the corpus counts.

Discard rule, per threshold t: byte-identical copies (same sha256, similarity 1) collapse to one file first. Then
every pair of distinct fonts with s > t is visited in random order; if both are still kept, one of the two, chosen
at random, is discarded. Averaged over --seeds random orders.

Finding the pairs: exact cosine over the upper triangle by blockwise matmul (GPU via torch if available, numpy
otherwise), keeping every pair above min(t). An SVD projection + faiss range search was tried first and dropped: at
384 dims its cosine error reached 0.29 (99.99th percentile), so the safety margin swallowed most of the corpus.

    python experiments/internet-archive/dedup.py dataset/internetArchive/cache/<zip stem> [more caches...]

Writes dataset/internetArchive/dedup/<name>/: results.json (the table), pairs.npz (all pairs above min(t)),
keep_t<t>.json (sha256 list kept at each threshold, seed 0), pairs_near_t.png (example pairs just above each t).
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image

_repoRoot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_repoRoot)

DEFAULT_THRESHOLDS = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
PLACEHOLDER_SIMILARITY = 0.999   # mean cosine between a font's own dedup letters; above = one box drawn for all


def interLetterSimilarity(dedup, rows, batch=4096):
    """Mean cosine between each font's own dedup glyphs. Fonts that map every letter to the same placeholder outline
    (a .notdef-style box that passes the cmap check) score ~1; real fonts, including letters-in-a-frame decorative
    ones, stay below ~0.995 (checked by eye on the Mega Pack chunk)."""
    out = np.empty(len(rows), np.float32)
    iu = np.triu_indices(dedup.shape[1], 1)
    for s in range(0, len(rows), batch):
        b = dedup[rows[s:s + batch]].reshape(len(rows[s:s + batch]), dedup.shape[1], -1).astype(np.float32)
        b /= np.linalg.norm(b, axis=2, keepdims=True) + 1e-8
        out[s:s + batch] = np.einsum("nid,njd->nij", b, b)[:, iu[0], iu[1]].mean(1)
    return out


def loadCaches(cacheDirs):
    """Merge caches by sha256. Returns per-unique-content rows (cache, row) and every file's content index."""
    contentOf, sources, fileContent, fileInfo = {}, [], [], []
    for cache in cacheDirs:
        with open(os.path.join(cache, "files.json"), encoding="utf8") as f:
            files = json.load(f)
        present = np.load(os.path.join(cache, "present.npy"))
        with open(os.path.join(cache, "meta.json"), encoding="utf8") as f:
            meta = json.load(f)
        dedupIndex = [meta["letters"].index(c) for c in meta["dedupLetters"]]
        comparable = present[:, dedupIndex].all(1)
        placeholder = np.zeros(len(present), bool)
        candidates = np.nonzero(comparable)[0]
        placeholder[candidates] = interLetterSimilarity(np.load(os.path.join(cache, "dedup.npy"), mmap_mode="r"),
                                                        candidates) > PLACEHOLDER_SIMILARITY
        for entry in files:
            if entry["row"] is None:
                fileContent.append(-2)            # unreadable file
            elif not comparable[entry["row"]]:
                fileContent.append(-1)            # no dedup glyphs (failed to open, or letters missing)
            elif placeholder[entry["row"]]:
                fileContent.append(-3)            # every letter drawn as the same placeholder box
            else:
                sha = entry["sha256"]
                if sha not in contentOf:
                    contentOf[sha] = len(sources)
                    sources.append((cache, entry["row"], meta["names"][entry["row"]], sha))
                fileContent.append(contentOf[sha])
            fileInfo.append((cache, entry["path"]))
    return sources, np.array(fileContent), fileInfo


def loadVectors(sources):
    arrays = {}
    out = None
    for i, (cache, row, _, _) in enumerate(sources):
        if cache not in arrays:
            arrays[cache] = np.load(os.path.join(cache, "dedup.npy"), mmap_mode="r")
        if out is None:
            out = np.lib.format.open_memmap(os.path.join(SCRATCH, "vectors.npy"), mode="w+", dtype=np.float16,
                                            shape=(len(sources), int(np.prod(arrays[cache].shape[1:]))))
        out[i] = arrays[cache][row].reshape(-1).astype(np.float32) / 255.0
    norms = np.empty(len(sources), np.float32)
    for s in range(0, len(sources), 8192):
        block = out[s:s + 8192].astype(np.float32)
        norms[s:s + 8192] = np.linalg.norm(block, axis=1)
        out[s:s + 8192] = block / np.maximum(norms[s:s + 8192, None], 1e-8)
    out.flush()
    return out


def exactPairs(vectors, threshold, rowBlock=4096, colBlock=16384):
    """Every pair (i < j) with cosine > threshold, exact, by blockwise matmul over the upper triangle. Uses a CUDA
    GPU through torch when available (fp16 storage, fp32 products), numpy BLAS otherwise."""
    n = len(vectors)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else None
    except ImportError:
        device = None
    allI, allJ, allS = [], [], []
    started = time.time()
    for r in range(0, n, rowBlock):
        rows = vectors[r:r + rowBlock].astype(np.float32)
        if device:
            rows = torch.from_numpy(rows).to(device)
        for c in range(r, n, colBlock):
            cols = vectors[c:c + colBlock].astype(np.float32)
            if device:
                block = (rows @ torch.from_numpy(cols).to(device).T).cpu().numpy()
            else:
                block = rows @ cols.T
            ii, jj = np.nonzero(block > threshold)
            keep = (jj + c) > (ii + r)
            ii, jj = ii[keep], jj[keep]
            allI.append((ii + r).astype(np.int32)); allJ.append((jj + c).astype(np.int32))
            allS.append(block[ii, jj])
        done = min(r + rowBlock, n)
        work = 1 - (1 - done / n) ** 2
        print(f"\r  exact pairs: rows {done}/{n}, pairs {sum(len(x) for x in allI)}, "
              f"~{(time.time() - started) * (1 - work) / max(work, 1e-9) / 60:.0f} min left", end="", flush=True)
    print()
    return np.concatenate(allI), np.concatenate(allJ), np.concatenate(allS)


def greedyDiscard(n, i, j, s, t, seed):
    rng = np.random.default_rng(seed)
    sel = np.nonzero(s > t)[0]
    order = rng.permutation(sel)
    coins = rng.random(len(order)) < 0.5
    alive = np.ones(n, bool)
    for e, coin in zip(order, coins):
        a, b = i[e], j[e]
        if alive[a] and alive[b]:
            alive[a if coin else b] = False
    return alive


def pairSheet(sources, i, j, s, thresholds, path, perThreshold=4, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {}
    rowsOut = []
    bounds = list(thresholds) + [1.0000001]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = np.nonzero((s > lo) & (s <= hi))[0]
        for e in rng.choice(sel, min(perThreshold, len(sel)), replace=False) if len(sel) else []:
            pair = []
            for k in (i[e], j[e]):
                cache, row = sources[k][0], sources[k][1]
                if cache not in arrays:
                    arrays[cache] = np.load(os.path.join(cache, "dedup.npy"), mmap_mode="r")
                pair.append(np.concatenate(list(arrays[cache][row]), 1))
            gap = np.full((48, 12), 0, np.uint8)
            rowsOut.append((np.concatenate([pair[0], gap, pair[1]], 1), f"s={s[e]:.4f} (t>{lo})",
                            sources[i[e]][2], sources[j[e]][2]))
    if not rowsOut:
        return
    from PIL import ImageDraw
    width = rowsOut[0][0].shape[1]
    sheet = Image.new("L", (width + 420, 52 * len(rowsOut)), 255)
    draw = ImageDraw.Draw(sheet)
    for r, (img, text, a, b) in enumerate(rowsOut):
        sheet.paste(Image.fromarray(255 - img), (0, 52 * r))
        draw.text((width + 8, 52 * r + 4), text, fill=0)
        draw.text((width + 8, 52 * r + 18), f"L: {a}"[:64], fill=0)
        draw.text((width + 8, 52 * r + 32), f"R: {b}"[:64], fill=0)
    sheet.save(path)


def main():
    global SCRATCH
    p = argparse.ArgumentParser()
    p.add_argument("caches", nargs="+")
    p.add_argument("--name", default=None)
    p.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()

    name = args.name or "+".join(os.path.basename(os.path.normpath(c)) for c in args.caches)
    out = os.path.join("dataset", "internetArchive", "dedup", name)
    os.makedirs(out, exist_ok=True)
    SCRATCH = out
    started = time.time()

    sources, fileContent, _ = loadCaches(args.caches)
    nFiles = len(fileContent)
    comparableFiles = int((fileContent >= 0).sum())
    nContent = len(sources)
    print(f"{nFiles} files; {comparableFiles} comparable (all of the dedup letters rendered); "
          f"{nContent} distinct by sha256", flush=True)

    vectors = loadVectors(sources)
    print(f"vectors loaded ({time.time() - started:.0f}s)", flush=True)
    i, j, s = exactPairs(vectors, min(args.thresholds))
    np.savez_compressed(os.path.join(out, "pairs.npz"), i=i, j=j, s=s,
                        sha256=np.array([x[3] for x in sources]))
    print(f"{len(i)} pairs with s > {min(args.thresholds)} ({time.time() - started:.0f}s)", flush=True)

    exactCopies = comparableFiles - nContent
    table = []
    for t in sorted(args.thresholds):
        kept = []
        for seed in range(args.seeds):
            alive = greedyDiscard(nContent, i, j, s, t, seed)
            kept.append(int(alive.sum()))
            if seed == 0:
                with open(os.path.join(out, f"keep_t{t}.json"), "w") as f:
                    json.dump([sources[k][3] for k in np.nonzero(alive)[0]], f)
        keptMean = float(np.mean(kept))
        table.append({"t": t, "pairsAbove": int((s > t).sum()), "kept": keptMean,
                      "keptMin": min(kept), "keptMax": max(kept),
                      "discarded": comparableFiles - keptMean,
                      "percentDiscarded": 100 * (comparableFiles - keptMean) / comparableFiles})

    pairSheet(sources, i, j, s, sorted(args.thresholds), os.path.join(out, "pairs_near_t.png"))
    results = {"caches": args.caches, "files": nFiles, "unreadableFiles": int((fileContent == -2).sum()),
               "notComparable": int((fileContent == -1).sum()), "placeholderGlyphs": int((fileContent == -3).sum()),
               "corpusBeforeDedup": comparableFiles,
               "distinctBySha256": nContent, "exactCopies": exactCopies, "seeds": args.seeds,
               "table": table, "seconds": time.time() - started}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    os.remove(os.path.join(out, "vectors.npy"))

    print(f"\nfiles {nFiles}: unreadable {results['unreadableFiles']}, "
          f"missing a dedup letter / failed to open {results['notComparable']}, "
          f"placeholder box for every letter {results['placeholderGlyphs']}")
    print(f"corpus before dedup: {comparableFiles}  (byte-identical copies among them: {exactCopies})\n")
    print("| t | pairs with s > t | discarded | % discarded | corpus after |")
    print("|---|---|---|---|---|")
    print(f"| (none) | | 0 | 0.0% | {comparableFiles} |")
    for row in table:
        print(f"| {row['t']} | {row['pairsAbove']} | {row['discarded']:.0f} | {row['percentDiscarded']:.1f}% | "
              f"{row['kept']:.0f} |")


if __name__ == "__main__":
    main()
