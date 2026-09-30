"""
Builds a glyph-bitmap cache at a NEW resolution in size-suffixed folders
("bitmaps<suffix>", "smallimage<suffix>") next to the existing ones, so the
original 48px cache is untouched. imageSize = int(fontSize * 1.5) everywhere in
this pipeline; fontSize=43 -> 64px.

Standalone by design: worker processes must NOT import the `utils` package
(its __init__ pulls in torch/transformers -- ~1 GB per worker on Windows spawn),
so standard.py is loaded directly by path and the MyFonts PNG conversion is an
inlined copy of utils.loaders.myfonts.loadRochesterImage (verify with --selfTest,
which reproduces existing dataset/smallimage bitmaps byte-for-byte at fontSize=32).

Resumable: re-running skips finished fonts (per-font "al.bmp" check) and stops
early on a completion marker; a marker is only written after a full pass.

    python -u experiments/embedding-geometry/build_bitmap_cache.py --fontSize 43 --cacheSuffix _64 --only standard
    python -u experiments/embedding-geometry/build_bitmap_cache.py --fontSize 43 --cacheSuffix _64 --only myfonts
"""
import argparse
import importlib.util
import os
import sys
import time
from glob import glob
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STANDARD_SOURCES = ["google", "dafont"]
MYFONTS_DIR = "dataset"

_standard = None


def loadStandard():
    """utils/loaders/standard.py loaded by path -- avoids executing utils/__init__.py."""
    global _standard
    if _standard is None:
        spec = importlib.util.spec_from_file_location(
            "standardLoader", os.path.join(REPO_ROOT, "utils", "loaders", "standard.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _standard = module
    return _standard


def renderFont(args):
    fontPath, fontSize, imageSize, directory, bitmapDir = args
    try:
        loadStandard().imagesFromFont(fontPath, fontSize, imageSize, directory, bitmapDir=bitmapDir)
        return None
    except Exception as e:
        return f"{fontPath!r} {e!r}"


def buildStandard(directory, fontSize, cacheSuffix, workers):
    standard = loadStandard()
    marker = standard.cacheCompleteMarker(directory, cacheSuffix)
    if os.path.exists(marker):
        print(f"{directory}: already complete ({marker})", flush=True)
        return

    bitmapDir = "bitmaps" + cacheSuffix
    for sub in (bitmapDir, "sdf" + cacheSuffix):
        os.makedirs(os.path.join(directory, sub), exist_ok=True)

    imageSize = int(fontSize * 1.5)
    fonts = (glob(os.path.join(directory, "fonts", "**", "*.ttf"), recursive=True)
             + glob(os.path.join(directory, "fonts", "**", "*.otf"), recursive=True))
    fonts = [f for f in fonts if os.path.isfile(f)]
    tasks = [(f, fontSize, imageSize, directory, bitmapDir) for f in fonts]
    print(f"{directory}: {len(tasks)} font files -> {directory}/{bitmapDir} at {imageSize}px, "
          f"{workers} workers", flush=True)

    start = time.time()
    errors = 0
    errorLog = os.path.join(REPO_ROOT, f"bitmap_cache_errors_{os.path.basename(directory)}{cacheSuffix}.log")
    with Pool(workers) as pool, open(errorLog, "a", encoding="utf-8") as log:
        for i, err in enumerate(pool.imap_unordered(renderFont, tasks, chunksize=8)):
            if err:
                errors += 1
                log.write(err.encode("ascii", "backslashreplace").decode("ascii") + "\n")
            if (i + 1) % 500 == 0 or i + 1 == len(tasks):
                rate = (i + 1) / (time.time() - start)
                print(f"{directory}: {i + 1}/{len(tasks)} fonts, {errors} errors, {rate:.1f} fonts/s, "
                      f"eta {(len(tasks) - i - 1) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    with open(marker, "w") as f:
        f.write("complete\n")
    print(f"{directory}: DONE in {(time.time() - start) / 60:.1f} min, {errors} font errors (see {errorLog})",
          flush=True)


# --- MyFonts (dataset/): inlined copy of utils.loaders.myfonts.loadRochesterImage -------------
def rochesterToCanvas(imagePath, fontSize):
    imageSize = int(fontSize * 1.5)
    padding = 8
    targetGlyphSize = imageSize - 2 * padding

    image = Image.open(imagePath).convert("RGB")
    array = np.array(image, dtype=np.float32)

    gray = 1.0 - (array[:, :, 0] / 255.0)

    col_max = np.max(gray, axis=0)
    nonzero_cols = np.where(col_max > 0.05)[0]
    if len(nonzero_cols) == 0:
        return None, None
    gray = gray[:, : nonzero_cols[-1] + 1]

    row_max = np.max(gray, axis=1)
    nonzero_rows = np.where(row_max > 0.05)[0]
    if len(nonzero_rows) == 0:
        return None, None
    gray = gray[nonzero_rows[0]: nonzero_rows[-1] + 1, :]

    if gray.shape[0] == 0 or gray.shape[1] == 0:
        return None, None

    h, w = gray.shape
    scale = min(targetGlyphSize / h, targetGlyphSize / w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(gray, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((imageSize, imageSize), dtype=np.float32)
    y0 = (imageSize - new_h) // 2
    x0 = (imageSize - new_w) // 2
    canvas[y0: y0 + new_h, x0: x0 + new_w] = resized

    name = os.path.basename(imagePath).removesuffix(".png")
    return name, canvas


def rochesterOutputName(name):
    fontName = name.split("_")[0]
    letter = name[-1].lower()
    case = "u" if name[-1] == name[-2] else "l"
    return f"{fontName} {letter}{case}.bmp"


def convertRochester(args):
    imagePath, fontSize, outDir = args
    try:
        name, array = rochesterToCanvas(imagePath, fontSize)
        if name is None:
            return 0
        img = Image.fromarray((array * 255).astype(np.uint8)).convert("L")
        img.save(os.path.join(outDir, rochesterOutputName(name)))
        return 1
    except Exception:
        return 0


def buildMyFonts(fontSize, cacheSuffix, workers):
    marker = os.path.join(MYFONTS_DIR, f"_cache_complete_smallimage{cacheSuffix}")
    if os.path.exists(marker):
        print(f"{MYFONTS_DIR}: already complete ({marker})", flush=True)
        return
    outDir = os.path.join(MYFONTS_DIR, "smallimage" + cacheSuffix)
    os.makedirs(outDir, exist_ok=True)

    pngs = glob(os.path.join(MYFONTS_DIR, "fontimage", "*.png"))
    tasks = [(p, fontSize, outDir) for p in pngs]
    print(f"{MYFONTS_DIR}: {len(tasks)} PNGs -> {outDir} at {int(fontSize * 1.5)}px, {workers} workers", flush=True)

    start = time.time()
    converted = 0
    with Pool(workers) as pool:
        for i, ok in enumerate(pool.imap_unordered(convertRochester, tasks, chunksize=500)):
            converted += ok
            if (i + 1) % 20000 == 0 or i + 1 == len(tasks):
                rate = (i + 1) / (time.time() - start)
                print(f"{MYFONTS_DIR}: {i + 1}/{len(tasks)} ({converted} written), {rate:.0f}/s, "
                      f"eta {(len(tasks) - i - 1) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    with open(marker, "w") as f:
        f.write("complete\n")
    print(f"{MYFONTS_DIR}: DONE in {(time.time() - start) / 60:.1f} min, {converted} bitmaps", flush=True)


def selfTest(count=25):
    """Inlined MyFonts conversion vs. the ORIGINAL pipeline's existing 48px bitmaps (fontSize=32)."""
    pngs = sorted(glob(os.path.join(MYFONTS_DIR, "fontimage", "*.png")))[::max(1, 40000)][:count]
    matched = checked = 0
    for path in pngs:
        name, array = rochesterToCanvas(path, 32)
        if name is None:
            continue
        existing = os.path.join(MYFONTS_DIR, "smallimage", rochesterOutputName(name))
        if not os.path.exists(existing):
            continue
        mine = np.array(Image.fromarray((array * 255).astype(np.uint8)).convert("L"))
        theirs = np.array(Image.open(existing))
        checked += 1
        matched += int(mine.shape == theirs.shape and np.array_equal(mine, theirs))
    print(f"self test: {matched}/{checked} inlined-conversion bitmaps identical to existing 48px cache", flush=True)
    return checked > 0 and matched == checked


def report(fontSize, cacheSuffix):
    for directory in STANDARD_SOURCES:
        folder = os.path.join(directory, "bitmaps" + cacheSuffix)
        files = os.listdir(folder) if os.path.isdir(folder) else []
        print(f"{folder}: {len(files)} files", flush=True)
        for f in files[:1]:
            print(f"   sample {f}: shape {np.array(Image.open(os.path.join(folder, f))).shape}", flush=True)
    folder = os.path.join(MYFONTS_DIR, "smallimage" + cacheSuffix)
    files = os.listdir(folder) if os.path.isdir(folder) else []
    print(f"{folder}: {len(files)} files", flush=True)
    for f in files[:1]:
        print(f"   sample {f}: shape {np.array(Image.open(os.path.join(folder, f))).shape}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fontSize", type=int, default=43)
    parser.add_argument("--cacheSuffix", default="_64")
    parser.add_argument("--only", choices=["standard", "myfonts", "all", "selftest", "report"], default="all")
    parser.add_argument("--workers", type=int, default=8, help="rendering workers for google/dafont")
    parser.add_argument("--myFontsWorkers", type=int, default=10)
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    if args.only == "selftest":
        sys.exit(0 if selfTest() else 1)
    if args.only == "report":
        report(args.fontSize, args.cacheSuffix)
        return
    if args.only in ("standard", "all"):
        for directory in STANDARD_SOURCES:
            buildStandard(directory, args.fontSize, args.cacheSuffix, args.workers)
    if args.only in ("myfonts", "all"):
        buildMyFonts(args.fontSize, args.cacheSuffix, args.myFontsWorkers)
    report(args.fontSize, args.cacheSuffix)


if __name__ == "__main__":
    main()
