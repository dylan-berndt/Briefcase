"""
Builds a lightweight, cached name->raw-font-file-path map for the
myFonts/google/dafont directories, WITHOUT going through
CombinedQueryData/collectFontSetPaths (which eagerly renders every
font's full glyph set to bitmap files -- extreme overkill for
trial_search.py, which only ever needs to render a handful of
specific fonts per session). Reads each font's own name metadata via
PIL (fast, no rendering) and caches the result to disk since scanning
~39k files still takes real time -- do it once, not on every launch.

Font name convention matches utils.loaders.standard.imagesFromFont:
f"{fontName} {fontStyle}" from PIL's ImageFont.getname(), the same
convention embeddings/all.json's keys use.

    python3 experiments/diffusion-searches/build_font_path_cache.py
"""
import json
import os
from glob import glob

from PIL import ImageFont

CACHE_PATH = "embeddings/fontFilePaths.json"
DIRECTORIES = [os.path.join("dataset", "fonts"), os.path.join("google", "fonts"),
                os.path.join("dafont", "fonts")]


def buildCache():
    pathMap = {}
    paths = []
    for directory in DIRECTORIES:
        paths.extend(glob(os.path.join(directory, "**", "*.ttf"), recursive=True))
        paths.extend(glob(os.path.join(directory, "**", "*.otf"), recursive=True))

    print(f"{len(paths)} font files found, reading name metadata...")
    for i, path in enumerate(paths):
        try:
            font = ImageFont.truetype(path, 32)
            fontName, fontStyle = font.getname()
            pathMap[f"{fontName} {fontStyle}"] = path
        except Exception:
            continue
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(paths)}")

    print(f"{len(pathMap)} fonts resolved")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(pathMap, f)
    print(f"saved to {CACHE_PATH}")
    return pathMap


def loadOrBuildCache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return buildCache()


if __name__ == "__main__":
    buildCache()
