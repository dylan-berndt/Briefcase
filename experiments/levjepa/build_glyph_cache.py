"""
Packs the 52 latin glyphs (a-z = 0-25, A-Z = 26-51) of every font into one uint8 memmap for the LeVJEPA-style
pretraining, with ALL sources standardized on the MyFonts pipeline:

  * MyFonts (dataset/fontimage PNGs): utils.loaders.myfonts.loadRochesterImage, then the same truncating
    uint8 conversion the .bmp cache goes through.
  * Google / DaFont (font files): a large render (200px) of each glyph, then the identical tight-crop +
    scale-to-fit step (experiments/critical-review/render.py::rochesterFromGray). Glyphs missing from the
    font's cmap are skipped (PIL would otherwise draw the .notdef box, which imagesFromFont also guards
    against).

Outputs in --out (default dataset/levjepa, gitignored):
    glyphs.npy   uint8 [fonts, 52, 48, 48]   (np.load(..., mmap_mode="r"))
    present.npy  bool  [fonts, 52]           False = glyph missing / unreadable (pixels are 0)
    meta.json    font names, sources, letters, font-file paths, presence statistics

Font names follow the existing pipeline (`"<family> <style>"` from PIL's getname(), or the MyFonts file stem),
so they line up with tags/descriptions/embeddings. Two font files with the same name: the first one (sorted
path order, .ttf before .otf) wins, as in imagesFromFont's "already rendered" check.

    python experiments/levjepa/build_glyph_cache.py --limit 300 --out dataset/levjepa_test   # quick check
    python experiments/levjepa/build_glyph_cache.py                                          # full build
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from glob import glob
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFont

_scriptDir = os.path.dirname(os.path.abspath(__file__))
_repoRoot = os.path.dirname(os.path.dirname(_scriptDir))
if _repoRoot not in sys.path:
    sys.path.insert(0, _repoRoot)

from utils.loaders.myfonts import loadRochesterImage

LETTERS = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FONT_SIZE = 32                                  # -> 48px canvas, 32px glyph box (matches the 48px caches)
IMAGE_SIZE = int(FONT_SIZE * 1.5)
RENDER_PX = 200                                 # render.myfontsStyleFromFont's default


def loadRochesterFromGray():
    spec = importlib.util.spec_from_file_location(
        "criticalReviewRender", os.path.join(_repoRoot, "experiments", "critical-review", "render.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rochesterFromGray


rochesterFromGray = loadRochesterFromGray()


def toUint8(canvas):
    return np.rint(canvas * 255).astype(np.uint8)


# ---------------------------------------------------------------- workers

def myFontsJob(args):
    index, directory, name = args
    glyphs = np.zeros((len(LETTERS), IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    present = np.zeros(len(LETTERS), bool)
    for i, ch in enumerate(LETTERS):
        # lowercase files are "<font>_a.png", uppercase "<font>_AA.png"
        path = os.path.join(directory, "fontimage", f"{name}_{ch}.png" if ch.islower() else f"{name}_{ch}{ch}.png")
        if not os.path.exists(path):
            continue
        try:
            _, canvas = loadRochesterImage((path, FONT_SIZE))
        except Exception:
            continue
        if canvas is None:
            continue
        glyphs[i] = (canvas * 255).astype(np.uint8)      # the truncating conversion of the .bmp round trip
        present[i] = True
    return index, glyphs, present


def fontNameJob(path):
    try:
        family, style = ImageFont.truetype(path, FONT_SIZE).getname()
    except Exception:
        return path, None
    return path, f"{family} {style}"


def fontFileJob(args):
    index, path = args
    from fontTools.ttLib import TTFont
    glyphs = np.zeros((len(LETTERS), IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    present = np.zeros(len(LETTERS), bool)
    try:
        font = ImageFont.truetype(path, RENDER_PX)
        ttf = TTFont(path, lazy=True)
        cmaps = [table.cmap for table in ttf["cmap"].tables]
    except Exception:
        return index, glyphs, present
    for i, ch in enumerate(LETTERS):
        if not any(ord(ch) in cmap for cmap in cmaps):
            continue
        try:
            mask = font.getmask(ch)
            if mask.size == (0, 0):
                continue
            gray = np.asarray(Image.Image()._new(mask), dtype=np.float32) / 255.0
            canvas = rochesterFromGray(gray)
        except Exception:
            continue
        if canvas is None:
            continue
        glyphs[i] = toUint8(canvas)
        present[i] = True
    return index, glyphs, present


# ---------------------------------------------------------------- driver

def preview(glyphs, present, names, sources, path, perSource=6, seed=0):
    rng = np.random.RandomState(seed)
    strips = []
    for source in ("myfonts", "dafont", "google"):
        idx = [i for i in range(len(names)) if sources[i] == source and present[i].all()]
        if not idx:
            continue
        for i in rng.choice(idx, min(perSource, len(idx)), replace=False):
            a = np.concatenate(list(glyphs[i, :26]), axis=1)
            b = np.concatenate(list(glyphs[i, 26:]), axis=1)
            strips.append(np.concatenate([a, b], axis=0))
            strips.append(np.zeros((6, a.shape[1]), np.uint8) + 60)
    Image.fromarray(np.concatenate(strips, axis=0)).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--myFonts", default="dataset")
    p.add_argument("--standard", nargs="*", default=["dafont", "google"])
    p.add_argument("--out", default=os.path.join("dataset", "levjepa"))
    p.add_argument("--limit", type=int, default=None, help="fonts per source (quick check)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    started = time.time()

    # ---- decide the row list: MyFonts names, then unique named font files per standard directory
    names, sources, jobs, fontPaths = [], [], [], []
    myFontNames = sorted({os.path.basename(f)[:-len("_a.png")]
                          for f in glob(os.path.join(args.myFonts, "fontimage", "*_a.png"))})
    if args.limit:
        myFontNames = myFontNames[:args.limit]
    for name in myFontNames:
        jobs.append(("myfonts", args.myFonts, name))
        names.append(name); sources.append("myfonts"); fontPaths.append(None)

    with Pool(args.workers) as pool:
        for directory in args.standard:
            tag = os.path.basename(os.path.normpath(directory))
            paths = sorted(glob(os.path.join(directory, "fonts", "**", "*.ttf"), recursive=True)) + \
                sorted(glob(os.path.join(directory, "fonts", "**", "*.otf"), recursive=True))
            seen, kept = set(), []
            for path, key in pool.imap(fontNameJob, paths, chunksize=64):
                if key is None or key in seen:
                    continue
                seen.add(key)
                kept.append((path, key))
                if args.limit and len(kept) >= args.limit:
                    break
            print(f"{tag}: {len(paths)} font files -> {len(kept)} unique named fonts", flush=True)
            for path, key in kept:
                jobs.append(("file", path, key))
                names.append(key); sources.append(tag); fontPaths.append(path)

        n = len(names)
        glyphs = np.lib.format.open_memmap(os.path.join(args.out, "glyphs.npy"), mode="w+", dtype=np.uint8,
                                           shape=(n, len(LETTERS), IMAGE_SIZE, IMAGE_SIZE))
        present = np.zeros((n, len(LETTERS)), bool)
        print(f"{n} fonts ({sum(s == 'myfonts' for s in sources)} MyFonts) -> "
              f"{n * len(LETTERS) * IMAGE_SIZE ** 2 / 2 ** 30:.2f} GiB", flush=True)

        myfontsTasks = [(i, j[1], j[2]) for i, j in enumerate(jobs) if j[0] == "myfonts"]
        fileTasks = [(i, j[1]) for i, j in enumerate(jobs) if j[0] == "file"]
        done = 0
        for results in (pool.imap_unordered(myFontsJob, myfontsTasks, chunksize=16),
                        pool.imap_unordered(fontFileJob, fileTasks, chunksize=8)):
            for index, g, pr in results:
                glyphs[index] = g
                present[index] = pr
                done += 1
                if done % 500 == 0:
                    rate = done / (time.time() - started)
                    print(f"\r{done}/{n} fonts  {rate:.1f} fonts/s", end="", flush=True)
        print()

    glyphs.flush()
    np.save(os.path.join(args.out, "present.npy"), present)
    stats = {}
    for source in sorted(set(sources)):
        rows = np.array([s == source for s in sources])
        stats[source] = {"fonts": int(rows.sum()),
                         "all52": int(present[rows].all(1).sum()),
                         "all26lower": int(present[rows][:, :26].all(1).sum()),
                         "meanGlyphsPresent": float(present[rows].sum(1).mean())}
    crossDuplicates = len(names) - len(set(names))
    meta = {"letters": LETTERS, "imageSize": IMAGE_SIZE, "fontSize": FONT_SIZE, "renderPx": RENDER_PX,
            "names": names, "sources": sources, "fontFiles": fontPaths, "stats": stats,
            "crossSourceDuplicateNames": crossDuplicates, "buildSeconds": time.time() - started}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf8") as f:
        json.dump(meta, f)
    preview(np.load(os.path.join(args.out, "glyphs.npy"), mmap_mode="r"), present, names, sources,
            os.path.join(args.out, "preview.png"))
    print(json.dumps(stats, indent=1))
    print(f"cross-source duplicate names: {crossDuplicates}; total {time.time() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
