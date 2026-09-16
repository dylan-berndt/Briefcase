# Learns a mapping from frozen font embeddings (embeddings/all.json) to a
# TF-IDF-style vector over the font's style vocabulary (tags + adjectives).
# Previously ran the ViT backbone forward at every training step
# (CombinedQueryData loading images live, ViTEmbedder wrapping the frozen
# backbone + trainable head); now trains purely on the precomputed
# embeddings, matching trainTextMLP.py's fast matrix-math setup -- no image
# loading, no ViT forward, epochs take seconds not minutes.
#
# Vocabulary filtering: evalDescriptorGrounding.py measured how visually
# coherent each descriptor actually is (do fonts sharing it cluster
# together in the embedding space, vs a random-sample baseline), as a
# z-score against that baseline's own noise. Descriptors below Z_CUTOFF are
# dropped from the target vocabulary entirely -- a large chunk of the raw
# tag/adjective vocabulary is generic filler ("many", "similar",
# "typographic") with no real visual grounding, and training a target that's
# substantially noise is exactly why the previous embedding->vocabulary
# attempt likely underperformed regardless of the model itself.
#
# Target construction: per font, TF (this font's tag weight, or 1.0 for a
# present adjective) x IDF (log(N / doc_freq)) over the filtered vocabulary,
# L2-normalized per font (standard tf-idf practice) so cosine similarity is
# the natural comparison -- which is also the training objective below.
#
# Metrics are chosen to be read at a glance, not cross-referenced against a
# BCE loss value with no intuitive scale:
# - median true-tag rank: rank the WHOLE filtered vocabulary by predicted
#   score for a font, take the median rank (1=best) among that font's
#   actually-true tags. Directly comparable to a stated chance baseline
#   (vocabSize/2) -- same "rank vs. corpus size" framing as trainTextMLP.py.
# - precision@k: of the model's top-k guessed tags for a font, what
#   fraction are actually true for it.
# - recall@10: of a font's actually-true tags, what fraction show up
#   somewhere in the model's top 10 guesses.

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pygtrie import CharTrie

from utils import Config, loadDescriptionsFromSource

EMBEDDINGS_PATH = os.path.join("embeddings", "all.json")
GROUNDING_PATH = os.path.join("results", "descriptorGrounding.json")

# Minimum z-score (see evalDescriptorGrounding.py) for a descriptor to be
# kept in the target vocabulary. None = no cutoff, use every descriptor
# that was measured (every descriptor with >=30 fonts, per
# MIN_FONTS_PER_DESCRIPTOR in evalDescriptorGrounding.py) regardless of
# grounding score, for a direct comparison against the filtered run.
Z_CUTOFF = None

CHECKPOINT_DIR = os.path.join("checkpoints", "retrieval", "embeddingHead")

TEST_FRACTION = 0.1
SEED = 1234

BATCH_SIZE = 256
HIDDEN_DIM = 512
DROPOUT = 0.15
# Filtered run peaked at epoch ~15-20 then overfit for the remaining 480
# epochs -- this task converges fast. 60 leaves margin past that without
# repeating the same mistake; bestEpoch tracking below reports the actual
# peak regardless, so overshooting isn't silently misleading anymore.
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4


def loadFontEmbeddings(path=EMBEDDINGS_PATH):
    with open(path, "r") as file:
        data = json.load(file)
    return {name: np.array(vector, dtype=np.float32) for name, vector in data.items()}


def loadVocab(path=GROUNDING_PATH, zCutoff=Z_CUTOFF):
    with open(path, "r") as file:
        results = json.load(file)
    if zCutoff is None:
        vocab = sorted(r["descriptor"] for r in results)
    else:
        vocab = sorted(r["descriptor"] for r in results if r["z"] > zCutoff)
    return vocab


def matchEmbeddingsToDescriptions(fontEmbeddings, descriptions):
    """Same longest-prefix trie match used throughout this project --
    embeddings/all.json is keyed by per-style-variant render name, the
    description source by base family name."""
    trie = CharTrie()
    for key in descriptions:
        trie[key] = key

    candidates = {}
    for embName, vector in fontEmbeddings.items():
        match = trie.longest_prefix(embName.strip())
        if match.key is None:
            continue
        candidates.setdefault(match.key, []).append((embName, vector))

    return {k: min(v, key=lambda p: len(p[0]))[1] for k, v in candidates.items()}


def buildTfidfTargets(descriptions, names, vocab):
    vocabIndex = {tag: i for i, tag in enumerate(vocab)}
    V = len(vocab)

    fontTags = {}
    docFreq = np.zeros(V)
    for name in names:
        desc = descriptions[name]
        tags = {}
        for tag, weight in desc.tags.items():
            if tag in vocabIndex:
                tags[tag] = max(tags.get(tag, 0.0), float(weight))
        for adjective in desc.adjectives:
            if adjective in vocabIndex:
                tags[adjective] = max(tags.get(adjective, 0.0), 1.0)
        fontTags[name] = tags
        for tag in tags:
            docFreq[vocabIndex[tag]] += 1

    idf = np.log(len(names) / np.maximum(docFreq, 1))

    targets = np.zeros((len(names), V), dtype=np.float32)
    for row, name in enumerate(names):
        for tag, weight in fontTags[name].items():
            targets[row, vocabIndex[tag]] = weight * idf[vocabIndex[tag]]

    return targets, idf


class RetrievalHead(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hiddenDim, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hiddenDim, outputDim),
        )

    def forward(self, x):
        return self.net(x)


def cosineLoss(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (1 - (pred * target).sum(dim=-1)).mean()


def sparsifyToTfidf(preds, idf, topK):
    """
    Zero every dimension outside each font's own top-K predicted values,
    then rescale the kept ones by idf -- rebuilds a sparse TF-IDF vector
    from the model's own top guesses, the same construction
    buildTfidfTargets used for the true per-font targets (binary presence
    x idf), instead of scoring against the model's dense raw output where
    every one of the ~1290 dimensions contributes something (the training
    targets were sparse -- median 7 true tags/font -- but nothing forces
    the model's output to be, so most of its mass outside the real top
    attributes is noise, not signal).
    """
    k = min(topK, preds.shape[1])
    _, topIdx = preds.topk(k, dim=1)
    idfTensor = torch.as_tensor(idf, dtype=preds.dtype, device=preds.device)
    sparse = torch.zeros_like(preds)
    sparse.scatter_(1, topIdx, idfTensor[topIdx])
    return sparse


@torch.no_grad()
def evaluate(model, embeddings, targets, idf=None, topK=None, ks=(5, 10)):
    """topK=None uses the model's raw dense output (original behavior).
    topK=N sparsifies to each font's top-N predicted dims first (see
    sparsifyToTfidf) -- requires idf. NOTE: medianTrueTagRank gets noisier
    under sparsification, since every zeroed dimension ties at rank in an
    arbitrary order; precision@k/recall@k stay well-defined regardless
    (they only ask whether a true tag lands in the top k), so those are
    the metrics that actually matter for this comparison."""
    model.eval()
    raw = model(embeddings)
    if topK is not None:
        raw = sparsifyToTfidf(raw, idf, topK)
    preds = F.normalize(raw, dim=-1)

    V = targets.shape[1]
    order = preds.argsort(dim=1, descending=True)
    rankOf = torch.empty(order.shape, dtype=torch.float32, device=preds.device)
    ranks = torch.arange(1, V + 1, dtype=torch.float32, device=preds.device).unsqueeze(0).expand_as(order)
    rankOf.scatter_(1, order, ranks)

    isTrue = targets > 0
    trueRanks = torch.where(isTrue, rankOf, torch.full_like(rankOf, float("nan")))
    perFontMedianRank = trueRanks.nanmedian(dim=1).values
    medianRank = perFontMedianRank[~perFontMedianRank.isnan()].median().item()

    results = {"medianTrueTagRank": medianRank, "chanceRank": V / 2}
    for k in ks:
        topk = preds.topk(min(k, V), dim=1).indices
        hits = isTrue.gather(1, topk).float()
        results[f"precision@{k}"] = hits.mean(dim=1).mean().item()
        trueCounts = isTrue.sum(dim=1).clamp(min=1).float()
        results[f"recall@{k}"] = (hits.sum(dim=1) / trueCounts).mean().item()
    return results


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading font embeddings ...")
    fontEmbeddings = loadFontEmbeddings()

    print("Loading descriptor grounding scores ...")
    vocab = loadVocab()
    cutoffLabel = "no cutoff (all measured descriptors)" if Z_CUTOFF is None else f"z > {Z_CUTOFF}"
    print(f"{len(vocab)} descriptors kept ({cutoffLabel})")

    print("Loading raw tag/adjective descriptions ...")
    config = Config().load(os.path.join("configs", "vit.json"))
    descriptions = loadDescriptionsFromSource(config.dataset)

    matched = matchEmbeddingsToDescriptions(fontEmbeddings, descriptions)
    print(f"{len(matched)} fonts matched to embeddings")

    print("Building TF-IDF targets ...")
    allNames = sorted(matched.keys())
    targets, idf = buildTfidfTargets(descriptions, allNames, vocab)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(os.path.join(CHECKPOINT_DIR, "idf.json"), "w") as f:
        # Same IDF used to build training targets, keyed by vocab term --
        # search-time query scoring should weight by this too (see
        # estimateSearchQuality.py), otherwise only the font side of the
        # dot product is TF-IDF and common attributes can swamp rare,
        # actually-discriminative ones when a query happens to touch both.
        json.dump({tag: float(idf[i]) for i, tag in enumerate(vocab)}, f)

    # Fonts with none of their tags surviving the vocab filter have an
    # all-zero target -- no signal to learn or evaluate against, drop them.
    validMask = targets.sum(axis=1) > 0
    names = [n for n, keep in zip(allNames, validMask) if keep]
    targets = targets[validMask]
    print(f"{len(names)}/{len(allNames)} fonts have >=1 surviving tag "
          f"(median {int(np.median((targets > 0).sum(axis=1)))} per font)")

    fontDim = len(next(iter(matched.values())))
    fontEmbeddingMatrix = torch.tensor(np.stack([matched[n] for n in names]), dtype=torch.float32)
    targetMatrix = torch.tensor(targets, dtype=torch.float32)

    shuffled = list(range(len(names)))
    random.shuffle(shuffled)
    testSize = int(len(shuffled) * TEST_FRACTION)
    testIdx, trainIdx = shuffled[:testSize], shuffled[testSize:]
    print(f"{len(trainIdx)} train fonts, {len(testIdx)} test fonts")

    model = RetrievalHead(fontDim, HIDDEN_DIM, len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    trainEmb = fontEmbeddingMatrix[trainIdx].to(device)
    trainTarget = targetMatrix[trainIdx].to(device)
    testEmb = fontEmbeddingMatrix[testIdx].to(device)
    testTarget = targetMatrix[testIdx].to(device)

    stepsPerEpoch = max(1, len(trainIdx) // BATCH_SIZE)

    bestEpoch, bestRank, bestMetrics = None, float("inf"), None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(trainIdx))
        totalLoss = 0.0

        for step in range(stepsPerEpoch):
            batchIdx = perm[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            if len(batchIdx) < 2:
                continue

            pred = model(trainEmb[batchIdx])
            loss = cosineLoss(pred, trainTarget[batchIdx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()

        m = evaluate(model, testEmb, testTarget)
        if m["medianTrueTagRank"] < bestRank:
            bestEpoch, bestRank, bestMetrics = epoch, m["medianTrueTagRank"], m
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "head.pt"))
            with open(os.path.join(CHECKPOINT_DIR, "vocab.json"), "w") as f:
                json.dump(vocab, f)

        print(f"epoch {epoch:>4}  loss {totalLoss / stepsPerEpoch:.4f}  "
              f"median true-tag rank {m['medianTrueTagRank']:>5.1f}/{m['chanceRank']:.0f} chance  "
              f"P@5 {m['precision@5']*100:>5.1f}%  P@10 {m['precision@10']*100:>5.1f}%  "
              f"R@10 {m['recall@10']*100:>5.1f}%")

    print(f"\nBEST epoch {bestEpoch}: median true-tag rank {bestMetrics['medianTrueTagRank']:.1f}/"
          f"{bestMetrics['chanceRank']:.0f} chance  P@5 {bestMetrics['precision@5']*100:.1f}%  "
          f"P@10 {bestMetrics['precision@10']*100:.1f}%  R@10 {bestMetrics['recall@10']*100:.1f}%")

    # Reload the actual best checkpoint (in-memory weights are from the
    # last epoch, which may be past the best point) and compare raw dense
    # scoring against top-K sparsified TF-IDF reconstruction at a few K,
    # on the SAME held-out set -- a direct test of whether thresholding the
    # model's own predictions helps, before touching the search demo at all.
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "head.pt"), map_location=device))
    print("\n=== raw dense output vs. top-K sparsified TF-IDF reconstruction (same best checkpoint) ===")
    raw = evaluate(model, testEmb, testTarget)
    print(f"  raw (no threshold)  P@5 {raw['precision@5']*100:>5.1f}%  P@10 {raw['precision@10']*100:>5.1f}%  "
          f"R@10 {raw['recall@10']*100:>5.1f}%")
    for k in (5, 10, 15, 20, 30, 50):
        m = evaluate(model, testEmb, testTarget, idf=idf, topK=k)
        print(f"  top-{k:<3}             P@5 {m['precision@5']*100:>5.1f}%  P@10 {m['precision@10']*100:>5.1f}%  "
              f"R@10 {m['recall@10']*100:>5.1f}%")


if __name__ == "__main__":
    main()
