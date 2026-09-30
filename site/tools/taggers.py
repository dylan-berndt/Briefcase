"""The model side of the search bundle: anything that turns a font's glyph images into one logit per tag.

An adapter is a class with a static `load(path, device)` returning an object with

    vocab      list of tag names, one per output column
    fontSize   glyph size the model was trained at (the glyph canvas is 1.5x that)
    logits(glyphs)   uint8 [B, 26, H, W] lowercase a-z, rendered the MyFonts way -> float [B, len(vocab)]

Register it in ADAPTERS and pass its name as --adapter to scoreFonts.py. Everything after scoring (the bundle, the
server) only sees logits and vocab, so a new model changes nothing downstream.
"""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO)


def newestRun(path, required):
    """path itself if it is a run directory, else its newest subdirectory that has every required file.
    Runs from `finetuneTags.py --maxFonts` (smoke tests) are not picked automatically."""
    if all(os.path.exists(os.path.join(path, name)) for name in required):
        return path
    runs = []
    for name in sorted(os.listdir(path)):
        run = os.path.join(path, name)
        if os.path.isdir(run) and all(os.path.exists(os.path.join(run, r)) for r in required):
            args = {}
            if os.path.exists(os.path.join(run, "metrics.json")):
                with open(os.path.join(run, "metrics.json")) as file:
                    args = json.load(file).get("args", {})
            if args.get("maxFonts") is None:
                runs.append(run)
    if not runs:
        raise FileNotFoundError(f"no finished run with {required} under {path}")
    return runs[-1]


class FineTuneTags:
    """A run of finetuneTags.py: ViT backbone (checkpoint.pt + config.json) plus the tag head (head.pt, vocab.json)."""
    REQUIRED = ("checkpoint.pt", "config.json", "head.pt", "vocab.json")

    def __init__(self, model, vocab, fontSize, device, run):
        self.model, self.vocab, self.fontSize, self.device, self.run = model, vocab, fontSize, device, run

    @staticmethod
    def load(path, device):
        import torch
        from utils.vit import ViT
        from finetuneTags import FontTagger

        run = newestRun(path, FineTuneTags.REQUIRED)
        vit, _ = ViT.load(run)
        with open(os.path.join(run, "vocab.json")) as file:
            vocab = json.load(file)
        head = torch.load(os.path.join(run, "head.pt"), map_location="cpu")
        model = FontTagger(vit, len(vocab), hidden=head["2.weight"].shape[0])
        model.head.load_state_dict(head)

        fontSize = 32
        if os.path.exists(os.path.join(run, "metrics.json")):
            with open(os.path.join(run, "metrics.json")) as file:
                fontSize = json.load(file).get("args", {}).get("fontSize", fontSize)
        return FineTuneTags(model.to(device).eval(), vocab, fontSize, device, run)

    def logits(self, glyphs):
        import torch
        with torch.no_grad():
            batch = torch.tensor(glyphs, device=self.device).float() / 255.0
            with torch.autocast(device_type=self.device, enabled=self.device == "cuda"):
                return self.model(batch).float().cpu().numpy()


ADAPTERS = {"finetuneTags": FineTuneTags}


def loadTagger(adapter, path, device):
    if adapter not in ADAPTERS:
        raise ValueError(f"unknown adapter {adapter!r}, choose from {sorted(ADAPTERS)}")
    return ADAPTERS[adapter].load(path, device)
