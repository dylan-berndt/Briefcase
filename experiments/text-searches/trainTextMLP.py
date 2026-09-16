# Quick experiment: can a small text MLP make font-search queries
# discriminative, using the *existing* contrastive font embeddings
# (embeddings/all.json) as a frozen target space instead of training a new
# CLIP-style model from scratch?
#
# The deployed corpus is fixed -- every font this will ever be asked about is
# already known at training time. So the real generalization question isn't
# "does this work on unseen fonts" (there are none, in production), it's
# "does this work on unseen *queries* for fonts it has already seen." This
# script trains on ALL matched fonts (no font-level holdout) and instead
# holds out half of each font's descriptions; evaluation embeds a font's
# held-out descriptions and ranks them against the full corpus of font
# embeddings (all matched fonts, since that corpus is fixed either way).
#
# Font embeddings are already known to be well descriptive of style (see
# README/CLAUDE.md) -- only the text side is trained here: a sentence
# transformer (frozen, pre-embedded once up front so epochs are just matrix
# math) -> a couple of Linear layers -> the same 512-dim space as
# embeddings/all.json.
#
# Loss: InfoNCE generalized by Rank-N-Contrast (Zha et al., NeurIPS 2023).
# Plain InfoNCE treats every non-matching font in the batch as an equally
# "wrong" negative, which is a bad fit for font style -- a stylistically
# close font is a much softer negative than a random one. RNC instead ranks
# every other font in the batch by how similar it already is to the true
# target *in the frozen font-embedding space*, and only contrasts a given
# font against others ranked equally-or-less similar. The exact-match font
# (similarity 1.0, always ranked first) recovers ordinary InfoNCE; the
# extra ranked terms softly pull in stylistically-similar fonts instead of
# repelling them.
#
# No sigreg, no MoCo queue -- deliberately minimal so training is fast
# enough to eyeball a held-out-query-vs-full-corpus median rank every epoch.

import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pygtrie import CharTrie

EMBEDDINGS_PATH = os.path.join("embeddings", "all.json")
QUERIES_PATH = os.path.join("results", "fontQueriesV2.json")

SENTENCE_MODEL = "all-mpnet-base-v2"
CACHE_PATH = os.path.join("embeddings", f"sentenceQueries_{SENTENCE_MODEL}.pkl")

# Same exclusion list as utils.querying.GENERIC_FONTS (copied rather than
# imported -- that module drags in spacy/cv2/transformers for no benefit here).
GENERIC_FONTS = {"noto", "unifont", "quivira", "symbola", "dejavu", "gnu unifont"}

# Fraction of each font's OWN descriptions held out as unseen queries -- not
# a font-level split, since the corpus is fixed and every font is "in-sample."
TEST_QUERY_FRACTION = 0.2
# Fixed random subsample of fonts whose held-out queries get evaluated each
# epoch (kept constant across epochs for a stable curve); always ranked
# against the full corpus regardless of this sample size.
EVAL_SAMPLE_SIZE = 3000
SEED = 1234

BATCH_SIZE = 256
HIDDEN_DIM = 1024
DROPOUT = 0.15
EPOCHS = 4000
LR = 1e-3
TEMPERATURE = 0.05

# i-Mix (Lee et al., ICLR 2021): mixup adapted to contrastive losses. Manufactures
# a continuum of new training points between real ones -- a way to get more out of
# a fixed set of labeled pairs without new data or touching the frozen backbone.
MIXUP_ALPHA = 0.4


def loadFontEmbeddings(path=EMBEDDINGS_PATH):
    with open(path, "r") as file:
        data = json.load(file)
    return {name: np.array(vector, dtype=np.float32) for name, vector in data.items()}


def loadDescriptions(path=QUERIES_PATH):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # v truthy check drops any font with an empty query list -- generateV2.py
    # now retries those instead of saving them, but older/in-progress output
    # files can still have a few from before that fix.
    return {k: v for k, v in data.items() if v and not any(k.lower().startswith(p) for p in GENERIC_FONTS)}


def matchEmbeddingsToDescriptions(fontEmbeddings, descriptions):
    """
    embeddings/all.json is keyed by per-style-variant render name (e.g.
    " Really Petshop Italic  Really Petshop Italic"); fontQueries.json is
    keyed by base family name ("Really Petshop"). Match with the same
    longest-prefix trie CombinedQueryData uses, then for families with
    multiple matching style variants keep the shortest-named one as the
    canonical "regular" embedding (style suffixes only ever add characters).
    """
    trie = CharTrie()
    for key in descriptions:
        trie[key] = key

    candidates = {}
    for embName, vector in fontEmbeddings.items():
        match = trie.longest_prefix(embName.strip())
        if match.key is None:
            continue
        candidates.setdefault(match.key, []).append((embName, vector))

    return {descKey: min(options, key=lambda pair: len(pair[0]))[1] for descKey, options in candidates.items()}


def encodeDescriptions(descriptions, names, cachePath=CACHE_PATH, modelName=SENTENCE_MODEL):
    if os.path.exists(cachePath):
        with open(cachePath, "rb") as file:
            cached = pickle.load(file)
        if all(name in cached for name in names):
            return cached

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(modelName)

    flatQueries = []
    spans = []
    for name in names:
        queries = descriptions[name]
        spans.append((len(flatQueries), len(flatQueries) + len(queries)))
        flatQueries.extend(queries)

    print(f"Embedding {len(flatQueries)} descriptions with {modelName} ...")
    vectors = model.encode(flatQueries, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

    encoded = {name: vectors[start:end] for name, (start, end) in zip(names, spans)}

    os.makedirs(os.path.dirname(cachePath), exist_ok=True)
    with open(cachePath, "wb") as file:
        pickle.dump(encoded, file)

    return encoded


class TextMLP(nn.Module):
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
            nn.Linear(hiddenDim, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hiddenDim, outputDim),
        )

    def forward(self, x):
        return self.net(x)


def rankNContrastLoss(sim, labelSim):
    """
    sim: [B, B] anchor-to-key similarity logits, already divided by temperature.
    labelSim: [B, B] continuous ground truth for "how similar should row i
    and column j be" -- here, cosine similarity between the two fonts' own
    frozen embeddings. For each anchor row, ranks keys by labelSim descending
    and, for every key j, contrasts it only against keys ranked
    equally-or-less similar than j (Rank-N-Contrast). Returns per-anchor loss.
    """
    order = labelSim.argsort(dim=1, descending=True)
    simSorted = torch.gather(sim, 1, order)
    denom = torch.logcumsumexp(simSorted.flip(dims=[1]), dim=1).flip(dims=[1])
    return (denom - simSorted).mean(dim=1)


def contrastiveLoss(textEmb, fontEmb, temperature=TEMPERATURE):
    textEmb = F.normalize(textEmb, dim=-1)
    fontEmb = F.normalize(fontEmb, dim=-1)

    labelSim = (fontEmb @ fontEmb.T).detach()
    simTextToFont = (textEmb @ fontEmb.T) / temperature
    simFontToText = simTextToFont.T

    lossTextAnchor = rankNContrastLoss(simTextToFont, labelSim)
    lossFontAnchor = rankNContrastLoss(simFontToText, labelSim)
    return 0.5 * (lossTextAnchor.mean() + lossFontAnchor.mean())


def sampleOneEach(names, sentenceCache, rng):
    sampled = np.stack([sentenceCache[name][rng.integers(len(sentenceCache[name]))] for name in names])
    return torch.tensor(sampled, dtype=torch.float32)


def splitQueryCache(sentenceCache, names, testFraction, rng):
    """
    Per-font split of a font's own cached description embeddings into a
    train half and a held-out "unseen query" half -- not a font-level split.
    Guarantees at least one description on each side.
    """
    trainCache, testCache = {}, {}
    for name in names:
        vectors = sentenceCache[name]
        order = rng.permutation(len(vectors))
        testCount = min(max(1, round(len(vectors) * testFraction)), len(vectors) - 1)
        testCache[name] = vectors[order[:testCount]]
        trainCache[name] = vectors[order[testCount:]]
    return trainCache, testCache


@torch.no_grad()
def promptAveragedQuery(model, names, sentenceCache, device):
    """
    Inference-only prompt averaging: embed *every* cached description for a
    font through the trained MLP (one batched forward pass over all of them,
    grouped by font via index_add_ rather than a per-font python loop),
    normalize each, average per font, and renormalize. Reduces single
    phrasing's word-choice noise without touching how training samples
    (training still draws one random description per font per step).
    """
    flatQueries = np.concatenate([sentenceCache[name] for name in names], axis=0)
    groupIndex = torch.tensor(
        [i for i, name in enumerate(names) for _ in range(len(sentenceCache[name]))],
        dtype=torch.long, device=device)

    textEmb = torch.tensor(flatQueries, dtype=torch.float32, device=device)
    projected = F.normalize(model(textEmb), dim=-1)

    fontDim = projected.shape[-1]
    sums = torch.zeros(len(names), fontDim, device=device).index_add_(0, groupIndex, projected)
    counts = torch.zeros(len(names), device=device).index_add_(0, groupIndex, torch.ones_like(groupIndex, dtype=torch.float32))
    return F.normalize(sums / counts.unsqueeze(1), dim=-1)


@torch.no_grad()
def medianRank(model, evalNames, queryCache, fontEmbeddingMatrix, nameIndex, device):
    """
    Rank of each evalNames font's own embedding among the FULL corpus
    (fontEmbeddingMatrix covers every matched font, not just evalNames) --
    the real deployed scenario is a query against the whole known corpus.
    queryCache determines which half of a font's descriptions the query is
    drawn from (held-out test half, or the train half for the "seen-style
    query" diagnostic); either way it's prompt-averaged (see
    promptAveragedQuery). Uses a target-vs-rest similarity count instead of a
    full argsort, since the corpus (the key side) can be much larger than
    the number of fonts being evaluated this epoch.
    """
    projected = promptAveragedQuery(model, evalNames, queryCache, device)
    keyEmb = F.normalize(fontEmbeddingMatrix.to(device), dim=-1)

    sims = projected @ keyEmb.T  # [E, N] -- E evaluated fonts x full N-font corpus
    targetIdx = torch.tensor([nameIndex[name] for name in evalNames], device=device)
    targetSim = sims.gather(1, targetIdx.unsqueeze(1))
    ranks = (sims > targetSim).sum(dim=1).float() + 1
    return ranks.median().item(), ranks.mean().item()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading font embeddings ...")
    fontEmbeddings = loadFontEmbeddings()
    print("Loading descriptions ...")
    descriptions = loadDescriptions()

    print("Matching description families to font embeddings ...")
    matched = matchEmbeddingsToDescriptions(fontEmbeddings, descriptions)
    names = sorted(matched.keys())
    print(f"{len(names)} fonts with both a description and an embedding")

    sentenceCache = encodeDescriptions(descriptions, names)
    trainCache, testCache = splitQueryCache(sentenceCache, names, TEST_QUERY_FRACTION, rng)

    fontDim = len(next(iter(matched.values())))
    fontEmbeddingMatrix = torch.tensor(np.stack([matched[name] for name in names]), dtype=torch.float32)
    nameIndex = {name: i for i, name in enumerate(names)}

    trainNames = names  # every font is in-sample -- the corpus is fixed
    evalNames = list(rng.choice(names, size=min(EVAL_SAMPLE_SIZE, len(names)), replace=False))
    print(f"{len(names)} fonts total (all used for training); "
          f"evaluating held-out queries for {len(evalNames)} of them each epoch, "
          f"ranked against the full {len(names)}-font corpus")

    textDim = next(iter(sentenceCache.values())).shape[-1]
    model = TextMLP(textDim, HIDDEN_DIM, fontDim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    stepsPerEpoch = max(1, len(trainNames) // BATCH_SIZE)

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(trainNames)
        totalLoss = 0.0

        for step in range(stepsPerEpoch):
            batchNames = trainNames[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            if len(batchNames) < 2:
                continue

            textEmb = sampleOneEach(batchNames, trainCache, rng).to(device)
            fontEmb = fontEmbeddingMatrix[[nameIndex[name] for name in batchNames]].to(device)

            # i-Mix: mix each anchor with a random in-batch partner, then blend the
            # two anchors' contrastive losses by the same interpolation weight --
            # exact since the RNC/InfoNCE loss is linear in which row is "the" target.
            lam = float(rng.beta(MIXUP_ALPHA, MIXUP_ALPHA)) if MIXUP_ALPHA > 0 else 1.0
            perm = torch.randperm(len(batchNames), device=device)

            mixedText = lam * textEmb + (1 - lam) * textEmb[perm]
            projected = model(mixedText)

            loss = lam * contrastiveLoss(projected, fontEmb) + (1 - lam) * contrastiveLoss(projected, fontEmb[perm])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

        model.eval()
        # held-out query: font's own description NOT seen during training this run
        heldOutMedian, heldOutMean = medianRank(model, evalNames, testCache, fontEmbeddingMatrix, nameIndex, device)
        # seen-style query diagnostic: same fonts, but averaged over their TRAIN-half
        # descriptions instead -- gap between this and held-out isolates query-phrasing
        # generalization from everything else (font itself is "in-sample" either way).
        seenMedian, seenMean = medianRank(model, evalNames, trainCache, fontEmbeddingMatrix, nameIndex, device)

        elapsed = time.time() - start
        print(f"epoch {epoch:>4}  ({elapsed:>7.1f}s)  loss {totalLoss / stepsPerEpoch:.4f}  "
              f"held-out query median rank {heldOutMedian:>6.1f}/{len(names)} (mean {heldOutMean:>7.1f})  "
              f"seen-style query median rank {seenMedian:>6.1f}/{len(names)} (mean {seenMean:>7.1f})")


if __name__ == "__main__":
    main()
