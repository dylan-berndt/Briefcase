"""
Embeds the font corpus with a checkpoint, using the same utils.embeddings.generateEmbeddings (and
font/letter set) that produced embeddings/all.json. The glyph path map is the 48px one from
embeddings/fontGlyphPaths.pkl with each path redirected to <folder><suffix>; fonts whose glyph is
missing there are dropped. suffix="" resolves to the original 48px bitmaps/smallimage folders.
checkpoint.pt is always a plain ViT state dict (see ProjectedViT/vitOf in retrain_strong_sigreg.py),
so this loads correctly whether or not the run used a projector head -- the projector itself is not
part of the embedding.

    python experiments/embedding-geometry/embed_strong_checkpoint.py <checkpointDir> <cacheName> [suffix]
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)
_os.chdir(_repoRoot)

import pickle

import numpy as np
import torch

from utils.vit import ViT
from utils.embeddings import generateEmbeddings

def main():
    checkpointDir, cacheName = _sys.argv[1], _sys.argv[2]
    suffix = _sys.argv[3] if len(_sys.argv) > 3 else "_64"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = ViT.load(checkpointDir)
    model.to(device).eval()
    print(f"checkpoint imageSize={config.model.imageSize} embedDim={config.model.embedDim} cacheSuffix={suffix!r}", flush=True)

    with open(_os.path.join("embeddings", "fontGlyphPaths.pkl"), "rb") as f:
        pathMap = pickle.load(f)

    names, letters, paths, missing = [], [], [], set()
    for name, letterMap in pathMap.items():
        for letter, path in letterMap.items():
            if not (letter.isalpha() and letter.islower() and len(letter) == 1):
                continue
            head, tail = _os.path.split(path)
            source, folder = _os.path.split(head)
            newPath = _os.path.join(source, folder + suffix, tail)
            if not _os.path.exists(newPath):
                missing.add(name)
                continue
            names.append(name); letters.append(letter); paths.append(newPath)
    print(f"{len(set(names))} fonts with glyphs, {len(missing)} fonts with some glyph missing in {suffix!r} cache", flush=True)

    fontData = {"names": np.array(names), "letters": np.array(letters), "paths": np.array(paths)}
    with torch.no_grad():
        embeddings = generateEmbeddings(fontData, model, fileName=cacheName)
    print(f"{len(embeddings)} embeddings saved to embeddings/{cacheName}.json")


if __name__ == "__main__":
    main()
