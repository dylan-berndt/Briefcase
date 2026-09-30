"""
Builds a cached name -> {letter: bitmapPath} map using the SAME loader
functions the rest of the repo's search UIs use for glyph rendering
(utils.loaders.myfonts.loadMyFontsImagePaths, utils.loaders.standard.
collectFontSetPaths -- exactly what utils.search.FontSearch._buildPathMap
builds from CombinedQueryData's own names/letters/paths). An earlier
version of this cache read font names via a raw PIL metadata scan of
.ttf/.otf files directly, avoiding these loaders entirely because
collectFontSetPaths eagerly opens (ImageFont.truetype + fontTools.TTFont)
every font FILE it finds even when its bitmaps are already cached -- but
that raw scan only matched ~54% of the corpus (google/dafont font FILES
don't always name themselves the way collectFontSetPaths's underlying
bitmap cache does, and it can't see the MyFonts dataset/ source at all).
Per-glyph bitmap rendering itself is properly cached to disk already
(imagesFromFont skips any bitmap that exists) -- it's the per-file
open+parse that's unavoidably slow here, so the fix is the same pattern
used everywhere else in this investigation: pay that cost ONCE and
cache the RESULT (this script), not skip the real pipeline for a
faster-but-incomplete substitute.

Uses the same dataset config (fontSize/maps/directories) that produced
embeddings/all.json, read directly from checkpoints/pretrain/best's own
config.json rather than hardcoded, so this cache stays in sync with
whatever corpus population that checkpoint was actually built from.

    python3 experiments/diffusion-searches/navigation_classifier/build_font_path_cache.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_baseDir = _os.path.dirname(_scriptDir)
_repoRoot = _os.path.dirname(_os.path.dirname(_baseDir))
for _p in [_repoRoot, _baseDir] + [_os.path.join(_baseDir, _d) for _d in _os.listdir(_baseDir)
                                     if _os.path.isdir(_os.path.join(_baseDir, _d))]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import os
import pickle

import numpy as np

from utils.loaders.myfonts import loadMyFontsImagePaths
from utils.loaders.standard import collectFontSetPaths

CACHE_PATH = os.path.join("embeddings", "fontGlyphPaths.pkl")
PRETRAIN_CONFIG_PATH = os.path.join("checkpoints", "pretrain", "best", "config.json")


def buildCache():
    with open(PRETRAIN_CONFIG_PATH) as f:
        datasetConfig = json.load(f)["dataset"]

    names, letters, paths = [], [], []

    myFontsDir = datasetConfig["directories"]["myFonts"]
    data = loadMyFontsImagePaths(myFontsDir, datasetConfig["fontSize"])
    names.append(data["names"]); letters.append(data["letters"]); paths.append(data["paths"])

    for directory in datasetConfig["directories"]["standard"]:
        data = collectFontSetPaths(directory, datasetConfig["fontSize"], datasetConfig["maps"])
        names.append(data["names"]); letters.append(data["letters"]); paths.append(data["paths"])

    names = np.concatenate(names, axis=0)
    letters = np.concatenate(letters, axis=0)
    paths = np.concatenate(paths, axis=0)

    pathMap = {}
    for name, letter, path in zip(names, letters, paths):
        pathMap.setdefault(name, {})[letter] = path

    print(f"{len(pathMap)} distinct fonts, {len(paths)} glyph paths total")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(pathMap, f)
    print(f"saved to {CACHE_PATH}")
    return pathMap


def loadOrBuildCache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return buildCache()


if __name__ == "__main__":
    pathMap = buildCache()

    from dataset import loadFontEmbeddings, EMBEDDINGS_PATH
    corpusNames = set(loadFontEmbeddings(EMBEDDINGS_PATH).keys())
    covered = sum(1 for n in corpusNames if n in pathMap)
    print(f"corpus coverage: {covered}/{len(corpusNames)} = {covered / len(corpusNames):.1%}")
