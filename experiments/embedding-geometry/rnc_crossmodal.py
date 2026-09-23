"""
A genuine cross-modal adapter test, not just a correlation number: trains a small two-tower system
(frozen visual backbone -> trainable MLP; frozen text representation -> trainable MLP, both projected
into a shared space) and measures REAL held-out retrieval recall@k by direct cosine ranking -- no
ASIF, no anchors, just "given a real held-out query, does the true font come back."

Runs a clean 2x2: {BGE dense embedding, LSA (TF-IDF+SVD)} text target x {RNC, InfoNCE} loss, same
architecture/steps/lr/data for all four. Both text representations are loaded from this project's own
EXISTING caches -- embeddings/sentenceQueries_BAAI_bge-large-en-v1.5.pkl and embeddings/tfidfFeatures_
d256.pkl, both already keyed to results/fontQueriesV2.json (18,940 fonts, 16 Gemma-generated queries
each, the newer/richer query set -- not the older fontQueries.json/8 Phi-4 queries used elsewhere this
session) -- rather than re-embedding or re-fitting either one from scratch.

One real query per font (index 0 of each cache/source, consistently) is used throughout, as both the
RNC ranking target's raw similarity source and the text tower's input.

    python experiments/embedding-geometry/rnc_crossmodal.py
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
import pickle

import numpy as np
import torch
import torch.nn as nn

VISUAL_SOURCE = "all_weak_sigreg_step85500"   # backbone of the newest/best pretrain-stage checkpoint
QUERIES_SOURCE = "fontQueriesV2"              # newer/richer query set (16 Gemma queries/font)
BGE_CACHE = "sentenceQueries_BAAI_bge-large-en-v1.5"
TFIDF_CACHE = "tfidfFeatures_d256"
HIDDEN_DIM = 512
SHARED_DIM = 128
TEMPERATURE = 0.04     # matches utils.training.EmbeddingLoss's convention, used for BOTH losses here
BATCH_SIZE = 256
STEPS = 4000
EVAL_EVERY = 1000
LR = 1e-3
SPLIT_SALT = "rnc-crossmodal-split-v1"
K_VALUES = [1, 5, 10, 50, 100]


def trainBucket(name, fraction=0.85):
    digest = hashlib.sha256(f"{SPLIT_SALT}:{name}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) < fraction


def normalizeNp(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class Tower(nn.Module):
    def __init__(self, inDim, hidden, outDim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inDim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, outDim),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


def rncLossOneWay(featSim, labelSim):
    """featSim, labelSim: [B, B], higher = more similar, anchors are ROWS. Not symmetric on its own --
    call twice (once on featSim, once on featSim.T) and average for a symmetric cross-modal loss,
    since labelSim is symmetric (raw text-text cosine) but featSim (text tower vs image tower output)
    is not. Unlike single-modality RNC, j=i is NOT excluded: text_i/image_i are different modalities of
    the SAME font, the genuine true pair (labelSim[i,i]=1, the maximum -- the InfoNCE "diagonal").

    Fully vectorized (no per-anchor Python loop): compare[i,j,k] = labelDist[i,k] >= labelDist[i,j],
    a [B,B,B] tensor -- fine at B=256 (67MB), and orders of magnitude faster than looping over anchors.
    """
    labelDist = -labelSim
    fs = featSim / TEMPERATURE
    compare = labelDist.unsqueeze(1) >= labelDist.unsqueeze(2)          # [B,B,B]: compare[i,j,k]
    fsBroadcast = fs.unsqueeze(1).expand(-1, fs.shape[0], -1)           # [B,B,B]: fsBroadcast[i,j,k]=fs[i,k]
    logDenom = torch.logsumexp(fsBroadcast.masked_fill(~compare, float("-inf")), dim=2)  # [B,B]
    lossMatrix = -fs + logDenom
    return lossMatrix.mean()


def rncLoss(featSim, labelSim):
    return 0.5 * (rncLossOneWay(featSim, labelSim) + rncLossOneWay(featSim.T, labelSim))


def infoNceLoss(featSim):
    B = featSim.shape[0]
    labels = torch.arange(B, device=featSim.device)
    logits = featSim / TEMPERATURE
    return 0.5 * (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels))


def evaluate(textTower, visualTower, textFeat, visualFeat, testIdx):
    textTower.eval(); visualTower.eval()
    with torch.no_grad():
        t = textTower(textFeat[testIdx]).cpu().numpy()
        v = visualTower(visualFeat[testIdx]).cpu().numpy()
    sims = t @ v.T
    order = np.argsort(-sims, axis=1)
    ranks = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(len(testIdx))])
    recall = {k: float(np.mean(ranks <= k)) for k in K_VALUES}
    textTower.train(); visualTower.train()
    return recall, float(np.median(ranks)), float(np.mean(ranks))


def runConfig(device, visual, textRaw, trainIdx, testIdx, lossName, targetName):
    torch.manual_seed(0)
    textTower = Tower(textRaw.shape[1], HIDDEN_DIM, SHARED_DIM).to(device)
    visualTower = Tower(visual.shape[1], HIDDEN_DIM, SHARED_DIM).to(device)
    opt = torch.optim.Adam(list(textTower.parameters()) + list(visualTower.parameters()), lr=LR)

    visualT = torch.tensor(visual, dtype=torch.float32, device=device)
    textT = torch.tensor(textRaw, dtype=torch.float32, device=device)
    textRawNorm = torch.tensor(normalizeNp(textRaw), dtype=torch.float32, device=device)

    rng = np.random.RandomState(0)
    history = []
    for step in range(1, STEPS + 1):
        batchIdx = rng.choice(trainIdx, BATCH_SIZE, replace=False)
        bi = torch.tensor(batchIdx, device=device)
        t = textTower(textT[bi])
        v = visualTower(visualT[bi])
        featSim = t @ v.T
        if lossName == "RNC":
            labelSim = textRawNorm[bi] @ textRawNorm[bi].T
            loss = rncLoss(featSim, labelSim)
        else:
            loss = infoNceLoss(featSim)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % EVAL_EVERY == 0:
            recall, medRank, meanRank = evaluate(textTower, visualTower, textT, visualT, testIdx)
            print(f"  [{targetName}/{lossName}] step {step:5d} loss={loss.item():.4f} "
                  f"recall@1={recall[1]*100:.2f}% recall@10={recall[10]*100:.2f}% "
                  f"recall@100={recall[100]*100:.2f}% medianRank={medRank:.0f}/{len(testIdx)}", flush=True)
            history.append((step, recall, medRank, meanRank))
    _, recall, medRank, meanRank = history[-1]
    return recall, medRank, meanRank


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    with open(_os.path.join("embeddings", f"{VISUAL_SOURCE}.json")) as f:
        visualAll = json.load(f)
    with open(_os.path.join("results", f"{QUERIES_SOURCE}.json")) as f:
        queriesAll = json.load(f)
    with open(_os.path.join("embeddings", f"{BGE_CACHE}.pkl"), "rb") as f:
        bgeAll = pickle.load(f)
    with open(_os.path.join("embeddings", f"{TFIDF_CACHE}.pkl"), "rb") as f:
        tfidfAll = pickle.load(f)
    # all four already share fontQueriesV2's keys/query-order; index 0 = the same real query everywhere
    names = sorted(set(visualAll) & set(queriesAll) & set(bgeAll) & set(tfidfAll))
    print(f"{len(names)} fonts with a visual embedding + cached BGE + cached LSA (all keyed to {QUERIES_SOURCE}.json)", flush=True)

    visual = np.array([visualAll[n] for n in names], dtype=np.float64)
    bgeVecs = np.array([bgeAll[n][0] for n in names], dtype=np.float64)
    lsaVecs = np.array([tfidfAll[n][0] for n in names], dtype=np.float64)
    print(f"BGE vectors: {bgeVecs.shape}   LSA vectors: {lsaVecs.shape}", flush=True)

    trainIdx = np.array([i for i, n in enumerate(names) if trainBucket(n)])
    testIdx = np.array([i for i, n in enumerate(names) if not trainBucket(n)])
    print(f"train={len(trainIdx)} test={len(testIdx)}", flush=True)

    results = {}
    for targetName, textRaw in [("BGE", bgeVecs.astype(np.float64)), ("LSA", lsaVecs.astype(np.float64))]:
        for lossName in ["RNC", "InfoNCE"]:
            print(f"\n=== target={targetName} loss={lossName} ===", flush=True)
            recall, medRank, meanRank = runConfig(device, visual, textRaw, trainIdx, testIdx, lossName, targetName)
            results[(targetName, lossName)] = (recall, medRank, meanRank)

    print(f"\n{'target':6s} {'loss':8s}" + "".join(f"  R@{k:<4d}" for k in K_VALUES) + "   medianRank   meanRank")
    for (targetName, lossName), (recall, medRank, meanRank) in results.items():
        print(f"{targetName:6s} {lossName:8s}" + "".join(f"  {recall[k]*100:5.2f}%" for k in K_VALUES) +
              f"   {medRank:6.0f}/{len(testIdx)}   {meanRank:6.0f}")


if __name__ == "__main__":
    main()
