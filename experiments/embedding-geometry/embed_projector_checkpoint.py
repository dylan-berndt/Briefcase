"""
Embeds the font corpus with a checkpoint's PROJECTOR output (not the backbone CLS feature) --
InfoNCE/SIGReg were trained on this space, not the one embed_strong_checkpoint.py uses. Same
glyph set / pooling (generateEmbeddings) as embeddings/all.json, so directly comparable.

    python experiments/embedding-geometry/embed_projector_checkpoint.py <checkpointDir> <cacheName> [suffix]
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)
if _scriptDir not in _sys.path:
    _sys.path.insert(0, _scriptDir)
_os.chdir(_repoRoot)

import pickle

import numpy as np
import torch
import torch.nn as nn

from utils.vit import ViT
from utils.embeddings import generateEmbeddings
from retrain_strong_sigreg import ProjectedViT


class ProjectorOnly(nn.Module):
    """generateEmbeddings expects `embedding = model(batch)` for any class not literally named
    "ViT" -- this returns just the projector output, discarding the backbone feature."""

    def __init__(self, projected):
        super().__init__()
        self.projected = projected

    def forward(self, x):
        _, z = self.projected(x)
        return z


def main():
    checkpointDir, cacheName = _sys.argv[1], _sys.argv[2]
    suffix = _sys.argv[3] if len(_sys.argv) > 3 else "_64"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vit, config = ViT.load(checkpointDir)
    state = torch.load(_os.path.join(checkpointDir, "train_state.pt"), map_location=device, weights_only=False)
    assert state.get("projDim", 0) > 0, f"{checkpointDir} has no saved projector"
    projected = ProjectedViT(vit, state["projDim"], 2048)  # PROJ_HIDDEN=2048 default, not itself saved
    projected.projector.load_state_dict(state["projector"])
    model = ProjectorOnly(projected).to(device).eval()
    print(f"checkpoint imageSize={config.model.imageSize} embedDim={config.model.embedDim} "
          f"projDim={state['projDim']} cacheSuffix={suffix!r}", flush=True)

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
