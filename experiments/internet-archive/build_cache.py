"""
Renders every font file extracted by fetch.py into two caches, one pass per unique file:

  * training glyphs: the 52 latin glyphs (a-z = 0-25, A-Z = 26-51) on the standardized MyFonts pipeline, identical
    to experiments/levjepa/build_glyph_cache.py's font-file path (200px render, glyphs missing from the cmap
    skipped, then render.py::rochesterFromGray tight crop + scale-to-fit into 32px of a 48px canvas). Same format as
    dataset/levjepa, so the two can be concatenated for pretraining.
  * dedup glyphs: DEDUP_LETTERS, each training glyph passed through centerAndScaleNormalize
    (experiments/embedding-geometry/glyph_normalize.py: re-centred on its ink centroid, ink bounding box scaled to
    60% of the canvas), so the same design rendered at a different size or offset lands on the same pixels.
    dedup.py compares fonts on these.

Byte-identical files (same sha256) are rendered once; files.json keeps every file and the row it maps to.

    python experiments/internet-archive/build_cache.py dataset/internetArchive/fonts/<zip stem>

Outputs in dataset/internetArchive/cache/<zip stem>/ (gitignored):
    files.json   every font file: path, size, sha256, row (index into the arrays below)
    glyphs.npy   uint8 [rows, 52, 48, 48]      training glyphs (np.load(..., mmap_mode="r"))
    present.npy  bool  [rows, 52]
    dedup.npy    uint8 [rows, len(DEDUP_LETTERS), 48, 48]
    meta.json    per row: font name ("<family> <style>" from PIL, as elsewhere), sha256, first path, error
Resumable: finished shards are recorded in progress.json.
"""
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from PIL import Image, ImageFont

_repoRoot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_repoRoot)

LETTERS = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEDUP_LETTERS = "aegknHQR"
IMAGE_SIZE = 48
RENDER_PX = 200
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")
SHARD = 1024


def loadFunction(path, name):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


rochesterFromGray = loadFunction(os.path.join("experiments", "critical-review", "render.py"), "rochesterFromGray")
centerAndScaleNormalize = loadFunction(os.path.join("experiments", "embedding-geometry", "glyph_normalize.py"),
                                       "centerAndScaleNormalize")
DEDUP_INDEX = [LETTERS.index(c) for c in DEDUP_LETTERS]


def toUint8(canvas):
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)


def hashJob(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return path, 0, None
    return path, len(data), hashlib.sha256(data).hexdigest()


def renderJob(path):
    from fontTools.ttLib import TTFont
    glyphs = np.zeros((len(LETTERS), IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    present = np.zeros(len(LETTERS), bool)
    dedup = np.zeros((len(DEDUP_LETTERS), IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    try:
        font = ImageFont.truetype(path, RENDER_PX)
        family, style = font.getname()
        ttf = TTFont(path, lazy=True, fontNumber=0)
        cmaps = [table.cmap for table in ttf["cmap"].tables]
    except Exception as e:
        return glyphs, present, dedup, None, f"{type(e).__name__}: {e}"[:200]
    for i, ch in enumerate(LETTERS):
        if not any(ord(ch) in cmap for cmap in cmaps):
            continue
        try:
            mask = font.getmask(ch)
            if mask.size == (0, 0):
                continue
            canvas = rochesterFromGray(np.asarray(Image.Image()._new(mask), dtype=np.float32) / 255.0)
        except Exception:
            continue
        if canvas is None:
            continue
        glyphs[i] = toUint8(canvas)
        present[i] = True
    for j, i in enumerate(DEDUP_INDEX):
        if present[i]:
            dedup[j] = toUint8(centerAndScaleNormalize(glyphs[i].astype(np.float64) / 255.0))
    return glyphs, present, dedup, f"{family} {style}", None


def isolatedRender(path, timeout=60):
    """One font in its own process, for shards where a font crashed or hung a pool worker."""
    with cf.ProcessPoolExecutor(1, mp_context=mp.get_context("fork")) as ex:
        try:
            return ex.submit(renderJob, path).result(timeout=timeout)
        except Exception as e:
            for proc in ex._processes.values():
                proc.kill()
            return (np.zeros((len(LETTERS), IMAGE_SIZE, IMAGE_SIZE), np.uint8), np.zeros(len(LETTERS), bool),
                    np.zeros((len(DEDUP_LETTERS), IMAGE_SIZE, IMAGE_SIZE), np.uint8), None,
                    f"crashed: {type(e).__name__}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fontDir")
    p.add_argument("--out", default=None)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--limit", type=int, default=None, help="first N font files (quick check)")
    args = p.parse_args()

    stem = os.path.basename(os.path.normpath(args.fontDir))
    out = args.out or os.path.join("dataset", "internetArchive", "cache", stem)
    os.makedirs(out, exist_ok=True)
    started = time.time()

    filesPath = os.path.join(out, "files.json")
    if os.path.exists(filesPath):
        with open(filesPath, encoding="utf8") as f:
            files = json.load(f)
        print(f"{len(files)} files from {filesPath}", flush=True)
    else:
        paths = sorted(os.path.join(d, n) for d, _, names in os.walk(args.fontDir) for n in names
                       if n.lower().endswith(FONT_EXTENSIONS))
        if args.limit:
            paths = paths[:args.limit]
        print(f"hashing {len(paths)} font files...", flush=True)
        with cf.ThreadPoolExecutor(8) as ex:
            hashed = list(ex.map(hashJob, paths, chunksize=64))
        rowOf, files = {}, []
        for path, size, sha in hashed:
            row = None if sha is None else rowOf.setdefault(sha, len(rowOf))
            files.append({"path": os.path.relpath(path, args.fontDir), "size": size, "sha256": sha, "row": row})
        with open(filesPath, "w", encoding="utf8") as f:
            json.dump(files, f)
        print(f"{len(files)} files, {len(rowOf)} unique by sha256 ({time.time() - started:.0f}s)", flush=True)

    firstPath = {}
    for entry in files:
        if entry["row"] is not None:
            firstPath.setdefault(entry["row"], os.path.join(args.fontDir, entry["path"]))
    rows = len(firstPath)
    rowPaths = [firstPath[r] for r in range(rows)]

    mode = "r+" if os.path.exists(os.path.join(out, "glyphs.npy")) else "w+"
    glyphs = np.lib.format.open_memmap(os.path.join(out, "glyphs.npy"), mode=mode, dtype=np.uint8,
                                       shape=(rows, len(LETTERS), IMAGE_SIZE, IMAGE_SIZE))
    dedup = np.lib.format.open_memmap(os.path.join(out, "dedup.npy"), mode=mode, dtype=np.uint8,
                                      shape=(rows, len(DEDUP_LETTERS), IMAGE_SIZE, IMAGE_SIZE))
    progressPath = os.path.join(out, "progress.json")
    progress = {"shards": [], "present": {}, "names": {}, "errors": {}}
    if os.path.exists(progressPath) and mode == "r+":
        with open(progressPath, encoding="utf8") as f:
            progress = json.load(f)
    doneShards = set(progress["shards"])

    shards = [s for s in range(0, rows, SHARD) if s not in doneShards]
    print(f"rendering {rows} unique fonts, {len(shards)} shards left, {args.workers} workers", flush=True)
    renderStart, renderedThisRun = time.time(), 0
    for start in shards:
        end = min(start + SHARD, rows)
        ex = cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork"))
        try:
            results = list(ex.map(renderJob, rowPaths[start:end], chunksize=8, timeout=1800))
            ex.shutdown()
        except (cf.process.BrokenProcessPool, cf.TimeoutError) as e:
            for proc in list(ex._processes.values()):
                proc.kill()
            ex.shutdown(wait=False, cancel_futures=True)
            print(f"\n  shard {start}: {type(e).__name__}, re-rendering one font per process", flush=True)
            results = [isolatedRender(path) for path in rowPaths[start:end]]
        for r, (g, pr, dd, name, error) in zip(range(start, end), results):
            glyphs[r], dedup[r] = g, dd
            progress["present"][str(r)] = "".join("1" if x else "0" for x in pr)
            progress["names"][str(r)] = name
            if error:
                progress["errors"][str(r)] = error
        glyphs.flush(); dedup.flush()
        progress["shards"].append(start)
        with open(progressPath + ".tmp", "w", encoding="utf8") as f:
            json.dump(progress, f)
        os.replace(progressPath + ".tmp", progressPath)
        renderedThisRun += end - start
        rate = renderedThisRun / (time.time() - renderStart)
        left = sum(min(s + SHARD, rows) - s for s in shards) - renderedThisRun
        print(f"\r  {len(progress['shards'])}/{-(-rows // SHARD)} shards  {rate:.1f} fonts/s  "
              f"~{left / max(rate, 1e-6) / 60:.0f} min left", end="", flush=True)
    print()

    present = np.array([[c == "1" for c in progress["present"][str(r)]] for r in range(rows)], bool)
    np.save(os.path.join(out, "present.npy"), present)
    meta = {"letters": LETTERS, "dedupLetters": DEDUP_LETTERS, "imageSize": IMAGE_SIZE, "renderPx": RENDER_PX,
            "fontDir": args.fontDir,
            "names": [progress["names"][str(r)] for r in range(rows)],
            "sha256": [None] * rows, "paths": rowPaths,
            "errors": {int(k): v for k, v in progress["errors"].items()}}
    for entry in files:
        if entry["row"] is not None:
            meta["sha256"][entry["row"]] = entry["sha256"]
    dedupPresent = present[:, DEDUP_INDEX].all(1)
    meta["stats"] = {"files": len(files), "unreadableFiles": sum(e["row"] is None for e in files),
                     "uniqueBySha256": rows, "openFailed": len(progress["errors"]),
                     "allDedupLetters": int(dedupPresent.sum()), "all52": int(present.all(1).sum()),
                     "atLeast26": int((present.sum(1) >= 26).sum()),
                     "seconds": time.time() - started}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf8") as f:
        json.dump(meta, f)
    print(json.dumps(meta["stats"], indent=1))


if __name__ == "__main__":
    main()
