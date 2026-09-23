"""
The check flagged as missing: does the ACTUALLY DEPLOYED finetuned model (ViTEmbedder visual tower
+ CLIPTextEmbedder text tower, checkpoints/finetune/2026-06-07 17-04) have a better raw visual<->text
similarity correlation than the pretrain-only backbone did? Uses embeddings/allText.json (already the
cached ViTEmbedder output) for the visual side and a LIVE forward pass of real per-font descriptions
(results/fontQueries.json, the LLM-generated queries CombinedQueryData actually trains against, not a
proxy encoder) through the real, trained CLIPTextEmbedder for the text side -- both halves of the
actual deployed pair, not a BGE stand-in.

    python experiments/embedding-geometry/deployed_correlation_check.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)
_os.chdir(_repoRoot)

import json

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

from utils.querying import CLIPTextEmbedder

CHECKPOINT = _os.path.join("checkpoints", "finetune", "2026-06-07 17-04", "ViT openai-clip-vit-base-patch32")
CLIP_NAME = "openai/clip-vit-base-patch32"


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    with open(_os.path.join("embeddings", "allText.json")) as f:
        visualAll = json.load(f)
    with open(_os.path.join("results", "fontQueries.json")) as f:
        queriesAll = json.load(f)
    names = sorted(set(visualAll) & set(queriesAll))
    print(f"{len(names)} fonts with both a deployed visual embedding and real LLM-generated queries", flush=True)

    # CLIPTextEmbedder.load() is broken (missing sharedDim arg) -- FontSearch.loadModels() builds/loads
    # it manually instead, so we do the same here.
    from utils.config import Config
    config = Config().load(_os.path.join(CHECKPOINT, "config.json"))
    textModel = CLIPTextEmbedder(CLIP_NAME, config.model.embedDim)
    textModel.load_state_dict(torch.load(_os.path.join(CHECKPOINT, "text.pt"), map_location=device, weights_only=False))
    textModel.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(CLIP_NAME)
    print(f"CLIPTextEmbedder loaded, sharedDim={config.model.embedDim}", flush=True)

    visual = np.array([visualAll[n] for n in names], dtype=np.float64)

    textVecs = []
    batchSize = 256
    with torch.no_grad():
        for start in range(0, len(names), batchSize):
            batchNames = names[start:start + batchSize]
            # one real query per font (the first LLM-generated one) -- a single genuine description,
            # not an average over several, so this is directly comparable to a real user's single query
            batchText = [queriesAll[n][0] for n in batchNames]
            tokens = tokenizer(batchText, padding=True, truncation=True, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items() if k != "token_type_ids"}
            out = textModel(tokens).cpu().numpy()
            textVecs.append(out)
            if (start // batchSize) % 10 == 0:
                print(f"\r{start + len(batchNames)}/{len(names)} queries embedded", end="", flush=True)
    print()
    text = np.concatenate(textVecs, axis=0).astype(np.float64)

    rng = np.random.RandomState(0)
    n = len(visual)
    i = rng.randint(0, n, 20000); j = rng.randint(0, n, 20000)
    mask = i != j; i, j = i[mask], j[mask]
    vCos = np.sum(normalize(visual)[i] * normalize(visual)[j], axis=1)
    tCos = np.sum(normalize(text)[i] * normalize(text)[j], axis=1)
    pear = pearsonr(vCos, tCos)
    spear = spearmanr(vCos, tCos)
    print(f"\nDEPLOYED MODEL (real ViTEmbedder visual / real CLIPTextEmbedder text, real descriptions):")
    print(f"  n_items={n}  Pearson={pear.statistic:+.4f}  Spearman={spear.statistic:+.4f}")
    print(f"  vCos[mean,std]=[{vCos.mean():.3f},{vCos.std():.3f}]  tCos[mean,std]=[{tCos.mean():.3f},{tCos.std():.3f}]")
    print(f"\n  for comparison, pretrain-only backbone (proxy BGE text): Pearson +0.134 / +0.157 (best / new checkpoint)")
    print(f"  for comparison, Flickr8k reference: Pearson +0.436")


if __name__ == "__main__":
    main()
