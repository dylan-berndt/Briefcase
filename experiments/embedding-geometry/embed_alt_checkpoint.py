"""
Re-embeds the font corpus with a DIFFERENT pretrain checkpoint (queueSize=
32768, batchSize=256 -- both larger than checkpoints/pretrain/best's
16384/128) that was trained but never promoted to best/latest, to check
whether it happens to produce a less-collapsed embedding space. Reuses
utils.embeddings.generateEmbeddings, the exact function that produced
embeddings/all.json, so the two are directly comparable (same pooling:
per-letter CLS token, L2-normalized then mean-pooled over lowercase a-z --
see utils/embeddings.py:51's documented pooling-order quirk).

    python3 experiments/embedding-geometry/embed_alt_checkpoint.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)

import pickle

import numpy as np
import torch

from utils.vit import ViT
from utils.embeddings import generateEmbeddings

CHECKPOINT_DIR = _os.path.join(_repoRoot, "checkpoints", "pretrain", "2026-06-14 23-38")
CACHE_NAME = "all_queue32768"
GLYPH_CACHE = _os.path.join(_repoRoot, "embeddings", "fontGlyphPaths.pkl")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print(f"loading checkpoint: {CHECKPOINT_DIR}")
    model, config = ViT.load(CHECKPOINT_DIR)
    model.to(device).eval()
    print(f"queueSize={config.queueSize} layers={config.model.layers} batchSize={config.batchSize}")

    with open(GLYPH_CACHE, "rb") as f:
        pathMap = pickle.load(f)

    names, letters, paths = [], [], []
    for name, letterMap in pathMap.items():
        for letter, path in letterMap.items():
            if letter.isalpha() and letter.islower() and len(letter) == 1:
                names.append(name)
                letters.append(letter)
                paths.append(path)

    fontData = {"names": np.array(names), "letters": np.array(letters), "paths": np.array(paths)}
    print(f"{len(set(names))} fonts, {len(names)} glyph paths")

    embeddings = generateEmbeddings(fontData, model, fileName=CACHE_NAME)
    print(f"{len(embeddings)} font embeddings saved to embeddings/{CACHE_NAME}.json")


if __name__ == "__main__":
    main()
