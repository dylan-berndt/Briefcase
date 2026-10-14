"""
Packs the 52 latin glyphs (a-z = 0-25, A-Z = 26-51) of every font into one uint8 memmap for the LeVJEPA-style
pretraining, with ALL sources standardized on the MyFonts pipeline:

  * MyFonts (dataset/fontimage PNGs): utils.loaders.myfonts.loadRochesterImage, then the same truncating
    uint8 conversion the .bmp cache goes through.
  * Font-file sources (dataset.standard: dafont, google, dataset/internetArchive): utils.loaders.standard.renderGlyph, the
    same render imagesFromFont writes into the utils bitmap caches (200px render, glyphs missing from the font's cmap
    skipped, then the shared tight-crop + scale-to-fit, standard.scaleToFit).

Then the near-duplicate step over the whole compiled cache (utils/dedup.py): dedup.npz holds every pair above
MIN_THRESHOLD and the placeholder-box fonts, and the thresholds-vs-discarded table is printed and saved to
dedup.json. Training picks a threshold with the config's dataset.dedupThreshold.

Outputs in dataset.glyphCache (dataset/levjepa, gitignored):
    glyphs.npy   uint8 [fonts, 52, 48, 48]   (np.load(..., mmap_mode="r"))
    present.npy  bool  [fonts, 52]           False = glyph missing / unreadable (pixels are 0)
    meta.json    font names, sources, letters, font-file paths, presence statistics
    dedup.npz, dedup.json

Font names follow the utils loaders: the MyFonts file stem, or standard.fontKey ("<family> <style> <sha256[:12]>")
for font files. Files with the same key (byte-identical copies, within or across sources) are packed once: the first
in source order, then sorted path order.

Sources (dataset.myFonts, dataset.standard) and the output directory (dataset.glyphCache) come from the config,
default configs/levjepa.json; directories without a fonts/ folder are skipped.

    python experiments/levjepa/build_glyph_cache.py --limit 300 --out dataset/levjepa_test   # quick check
    python experiments/levjepa/build_glyph_cache.py                                          # full build
"""
import argparse
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

from utils.dedup import NearDuplicates
from utils.loaders.myfonts import loadRochesterImage
from utils.loaders.standard import RENDER_PX, fontFilePaths, fontKey, openFont, renderGlyph, toUint8

LETTERS = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FONT_SIZE = 32                                  # -> 48px canvas, 32px glyph box (matches the 48px caches)
IMAGE_SIZE = int(FONT_SIZE * 1.5)
OUTPUT_FILES = ("glyphs.npy", "present.npy", "meta.json", "preview.png", "dedup.npz", "dedup.json",
                "dedup.npz.vectors.npy")


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


def fontKeyJob(path):
    try:
        font, _ = openFont(path)
        return path, fontKey(font, path)
    except Exception:
        return path, None


def fontFileJob(args):
    index, path = args
    glyphs = np.zeros((len(LETTERS), IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    present = np.zeros(len(LETTERS), bool)
    try:
        font, cmaps = openFont(path)
    except Exception:
        return index, glyphs, present
    for i, ch in enumerate(LETTERS):
        try:
            canvas = renderGlyph(font, cmaps, ch, FONT_SIZE)
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
    for source in sorted(set(sources)):
        idx = [i for i in range(len(names)) if sources[i] == source and present[i].all()]
        if not idx:
            continue
        for i in rng.choice(idx, min(perSource, len(idx)), replace=False):
            a = np.concatenate(list(glyphs[i, :26]), axis=1)
            b = np.concatenate(list(glyphs[i, 26:]), axis=1)
            strips.append(np.concatenate([a, b], axis=0))
            strips.append(np.zeros((6, a.shape[1]), np.uint8) + 60)
    if strips:
        Image.fromarray(np.concatenate(strips, axis=0)).save(path)


def main():
    p = argparse.ArgumentParser(description="sources and output dir come from the config's dataset section")
    p.add_argument("--config", default=os.path.join("configs", "levjepa.json"))
    p.add_argument("--out", default=None, help="override the config's dataset.glyphCache (quick checks)")
    p.add_argument("--limit", type=int, default=None, help="fonts per source (quick check)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--clean", action="store_true", help="delete this script's outputs in the cache dir first")
    args = p.parse_args()
    with open(args.config, encoding="utf8") as f:
        dataset = json.load(f)["dataset"]
    args.myFonts, args.standard = dataset["myFonts"], dataset["standard"]
    args.out = args.out or dataset["glyphCache"]

    os.makedirs(args.out, exist_ok=True)
    started = time.time()
    # only this script's own files are removed, never the directory: a mistyped glyphCache can't wipe a dataset
    stale = OUTPUT_FILES if args.clean else ("dedup.npz",)   # dedup pairs always follow a fresh build
    for name in stale:
        if os.path.exists(os.path.join(args.out, name)):
            os.remove(os.path.join(args.out, name))
            print(f"removed {os.path.join(args.out, name)}", flush=True)

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
        seen = set()
        for directory in args.standard:
            if not os.path.isdir(os.path.join(directory, "fonts")):
                print(f"{directory}: no fonts/ folder, skipped", flush=True)
                continue
            tag = os.path.basename(os.path.normpath(directory))
            paths = fontFilePaths(directory)
            kept, unreadable, repeated = [], 0, 0
            for path, key in pool.imap(fontKeyJob, paths, chunksize=64):
                if key is None:
                    unreadable += 1
                    continue
                if key in seen:
                    repeated += 1
                    continue
                seen.add(key)
                kept.append((path, key))
                if args.limit and len(kept) >= args.limit:
                    break
            print(f"{tag}: {len(paths)} font files -> {len(kept)} fonts ({repeated} byte-identical repeats, "
                  f"{unreadable} unreadable)", flush=True)
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

    # near-duplicates over the whole compiled corpus, after every source is in the cache
    table = NearDuplicates.fromGlyphCache(args.out).printTable(preferred=NearDuplicates.glyphCacheMyFonts(meta))
    with open(os.path.join(args.out, "dedup.json"), "w", encoding="utf8") as f:
        json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
