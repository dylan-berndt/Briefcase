"""
Rank-N-Contrast (Zha et al., NeurIPS 2023) adapter: a small MLP on top of the FROZEN pretrain
backbone, trained against a CONTINUOUS text-similarity target (TF-IDF cosine over each font's real
LLM-generated queries, results/fontQueries.json -- the same source utils.training.buildFalseNegativeMask
already builds a TF-IDF matrix from, live-built here the same way) instead of binary contrastive
same/different. RNC's loss only enforces that the adapter's own similarity ordering matches the
target's ordering -- it never demands two genuinely-similar-but-distinct fonts be pushed apart, which
plain InfoNCE (and every training run this investigation tried) does implicitly.

RNC loss, per anchor i and candidate j != i: everything AT LEAST as far from i as j (in TF-IDF
distance) forms the denominator of a softmax over feature similarity; only things closer than j are
excluded. This is the "rank" part -- no fixed positive/negative split, just consistent ordering.

Reports, before and after training, the SAME raw pairwise Pearson/Spearman correlation used throughout
this investigation (visual_text_correlation.py), now against the TF-IDF target specifically, on a
held-out font split -- so the adapter's real effect is measured the same way the problem was diagnosed.

    python experiments/embedding-geometry/rnc_adapter.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)
_os.chdir(_repoRoot)

import hashlib
import json

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

VISUAL_SOURCE = "all_weak_sigreg_step85500"   # backbone (not projector) of the best pretrain-stage checkpoint on disk -- apples-to-apples with all.json, which is also a backbone CLS feature, not a projector output
HIDDEN_DIM = 512
OUTPUT_DIM = 128
TEMPERATURE = 0.1
BATCH_SIZE = 256
STEPS = 3000
EVAL_EVERY = 250
LR = 1e-3
SPLIT_SALT = "rnc-adapter-split-v1"   # same hash-split idea as utils.pretraining.PairedImageData.split


def trainBucket(name, fraction=0.85):
    digest = hashlib.sha256(f"{SPLIT_SALT}:{name}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) < fraction


class Adapter(nn.Module):
    def __init__(self, inDim, hidden, outDim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inDim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, outDim),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


def rncLoss(featSim, labelSim):
    """featSim, labelSim: [B, B], higher = more similar. Diagonal is excluded per-row inside."""
    B = featSim.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=featSim.device)
    labelDist = -labelSim  # RNC ranks by distance; bigger labelDist = farther
    total = 0.0
    count = 0
    for i in range(B):
        others = ~eye[i]
        ld_i = labelDist[i][others]          # [B-1]
        fs_i = featSim[i][others] / TEMPERATURE  # [B-1]
        # compare[j,k] = True if ld_i[k] >= ld_i[j]  (k belongs to j's denominator)
        compare = ld_i.unsqueeze(0) >= ld_i.unsqueeze(1)   # [B-1, B-1]
        logDenom = torch.logsumexp(fs_i.unsqueeze(0).masked_fill(~compare, float("-inf")), dim=1)
        lossRow = -fs_i + logDenom
        total = total + lossRow.sum()
        count += lossRow.numel()
    return total / count


def correlation(vecs, sims, idx, seed=0):
    rng = np.random.RandomState(seed)
    n = len(idx)
    a = rng.randint(0, n, 20000); b = rng.randint(0, n, 20000)
    mask = a != b; a, b = a[mask], b[mask]
    ia, ib = idx[a], idx[b]
    vNorm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    vCos = np.sum(vNorm[a] * vNorm[b], axis=1)
    tCos = np.asarray(sims[ia, ib]).flatten()
    return pearsonr(vCos, tCos).statistic, spearmanr(vCos, tCos).statistic


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    with open(_os.path.join("embeddings", f"{VISUAL_SOURCE}.json")) as f:
        visualAll = json.load(f)
    with open(_os.path.join("results", "fontQueries.json")) as f:
        queriesAll = json.load(f)
    names = sorted(set(visualAll) & set(queriesAll))
    print(f"{len(names)} fonts with both a visual embedding and real LLM-generated queries", flush=True)

    docs = [" ".join(queriesAll[n]) for n in names]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", max_features=50000)
    tfidf = vectorizer.fit_transform(docs)  # L2-normalized rows -> tfidf @ tfidf.T is cosine
    print(f"TF-IDF matrix built: {tfidf.shape}", flush=True)

    visual = np.array([visualAll[n] for n in names], dtype=np.float64)
    trainIdx = np.array([i for i, n in enumerate(names) if trainBucket(n)])
    testIdx = np.array([i for i, n in enumerate(names) if not trainBucket(n)])
    print(f"train={len(trainIdx)} test={len(testIdx)}", flush=True)

    visualT = torch.tensor(visual, dtype=torch.float32, device=device)
    adapter = Adapter(visual.shape[1], HIDDEN_DIM, OUTPUT_DIM).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=LR)

    # baseline correlation, RAW frozen backbone, against the SAME TF-IDF target (apples-to-apples,
    # unlike the earlier BGE-embedding-based correlation numbers)
    pear0, spear0 = correlation(visual[testIdx], tfidf, testIdx)
    print(f"\nBEFORE (raw frozen backbone vs TF-IDF target, held-out): Pearson={pear0:+.4f} Spearman={spear0:+.4f}\n", flush=True)

    rng = np.random.RandomState(0)
    for step in range(1, STEPS + 1):
        batchIdx = rng.choice(trainIdx, BATCH_SIZE, replace=False)
        x = visualT[batchIdx]
        z = adapter(x)
        featSim = z @ z.T
        labelBlock = tfidf[batchIdx]
        labelSim = torch.tensor((labelBlock @ labelBlock.T).toarray(), dtype=torch.float32, device=device)
        loss = rncLoss(featSim, labelSim)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            adapter.eval()
            with torch.no_grad():
                zTest = adapter(visualT[testIdx]).cpu().numpy()
            pear, spear = correlation(zTest, tfidf, testIdx)
            print(f"step {step:5d}  train RNC loss={loss.item():.4f}  held-out Pearson={pear:+.4f} Spearman={spear:+.4f}", flush=True)
            adapter.train()

    torch.save(adapter.state_dict(), _os.path.join("experiments", "embedding-geometry", "rnc_adapter.pt"))
    print("\nsaved experiments/embedding-geometry/rnc_adapter.pt", flush=True)
    print(f"SUMMARY: before Pearson={pear0:+.4f}  after Pearson={pear:+.4f}  (Flickr reference: +0.436)")


if __name__ == "__main__":
    main()
