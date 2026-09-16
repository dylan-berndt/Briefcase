# Measures whether the trained CLIP-style contrastive text/image model
# (checkpoints/finetune, the one FontSearch uses) could support a hard
# text-narrowing pre-filter before grid-feedback search -- e.g. "cut 39,421
# fonts down to the top ~4,000 by text score, then grid-search inside that".
#
# The relevant number isn't R@1 (how the checkpoint was likely benchmarked
# before) -- it's recall@10%: for a real font's own description used as a
# proxy query, does that font's own image embedding land in the top ~4,000
# of the full corpus by text-image score? If not, a hard filter would throw
# the user's actual target away before grid search even starts.
#
# Uses each font's real Description.sample() (the same query format the
# contrastive model was trained on) as the query, and checks the percentile
# rank of that font's own cached image embedding (embeddings/allText.json)
# under FontSearch.encodeQuery.

import os
import random
import types

import numpy as np

from utils.search import FontSearch
from utils.querying import loadDescriptionsFromSource

SAMPLE_SIZE = 300
NARROW_FRACTIONS = [0.05, 0.10, 0.20, 0.30]

# The finetune checkpoint's backbone was 8 layers/embedDim 512
# (checkpoints/pretrain/best), not whatever checkpoints/pretrain/latest has
# grown to since (12 layers) -- FontSearch's DEFAULT_BACKBONE has drifted out
# of sync with DEFAULT_FINETUNE, so it must be passed explicitly here.
BACKBONE = os.path.join("checkpoints", "pretrain", "best")


def main():
    print("Loading FontSearch (finetuned CLIP-style image/text model) ...")
    dummyDataset = types.SimpleNamespace(names=np.array([]), letters=np.array([]), paths=np.array([]))
    search = FontSearch(backbone=BACKBONE, dataset=dummyDataset)
    print(f"Loaded {len(search.embeddings)} cached image embeddings")

    print("Loading real font descriptions ...")
    descriptions = loadDescriptionsFromSource(search.datasetConfig)
    print(f"Loaded {len(descriptions)} descriptions")

    overlap = sorted(set(descriptions.keys()) & set(search.embeddings.keys()))
    print(f"Overlap with cached embeddings: {len(overlap)} fonts")

    if not overlap:
        print("No overlap between description names and embedding names -- naming mismatch, aborting.")
        return

    random.seed(0)
    sample = random.sample(overlap, min(SAMPLE_SIZE, len(overlap)))

    keys = list(search.embeddings.keys())
    keyIndex = {name: i for i, name in enumerate(keys)}
    N = len(keys)

    ranks = []
    for i, name in enumerate(sample):
        query = descriptions[name].sample()
        scores = search.encodeQuery(query)
        order = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        orderedNames = [n for n, _ in order]
        rank = orderedNames.index(name)  # 0 = perfect (top-1)
        ranks.append(rank)
        print(f"\r{i + 1}/{len(sample)}  last: '{query[:50]}' -> rank {rank}/{N}", end="")
    print()

    ranks = np.array(ranks)
    print(f"\nN={len(ranks)} fonts, corpus size={N}")
    print(f"median rank: {np.median(ranks):.0f}  (percentile {np.median(ranks)/N*100:.1f}%)")
    print(f"mean rank: {ranks.mean():.0f}  (percentile {ranks.mean()/N*100:.1f}%)")
    print(f"R@1: {(ranks == 0).mean()*100:.1f}%")
    print(f"R@10: {(ranks < 10).mean()*100:.1f}%")
    print(f"R@100: {(ranks < 100).mean()*100:.1f}%")

    print("\n=== Recall at narrowing cutoffs ===")
    for frac in NARROW_FRACTIONS:
        cutoff = int(N * frac)
        recall = (ranks < cutoff).mean() * 100
        print(f"cutoff={cutoff:>6} ({frac*100:>4.0f}% of corpus): recall={recall:5.1f}%  "
              f"(i.e. {100-recall:.1f}% of targets would be thrown away)")


if __name__ == "__main__":
    main()
