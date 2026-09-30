"""A randomly initialised finetuneTags run directory, small enough to load in a test."""

import json
import os

import torch

from finetuneTags import FontTagger
from utils.config import Config
from utils.vit import ViT

VOCAB = ["bold", "thin", "serif", "script", "horror", "zebra-stripe"]


def makeRun(path, vocab=VOCAB, hidden=8, fontSize=32, seed=0, smoke=False):
    os.makedirs(path, exist_ok=True)
    torch.manual_seed(seed)
    config = Config()
    config["model"] = Config(layers=1, imageSize=int(fontSize * 1.5), patchSize=8, embedDim=16, heads=2)
    config.save(os.path.join(path, "config.json"))
    vit = ViT(config.model)
    model = FontTagger(vit, len(vocab), hidden)
    torch.save(vit.state_dict(), os.path.join(path, "checkpoint.pt"))
    torch.save(model.head.state_dict(), os.path.join(path, "head.pt"))
    with open(os.path.join(path, "vocab.json"), "w") as file:
        json.dump(vocab, file)
    with open(os.path.join(path, "metrics.json"), "w") as file:
        json.dump({"args": {"fontSize": fontSize, "maxFonts": 300 if smoke else None, "hidden": hidden}}, file)
    return str(path), model
