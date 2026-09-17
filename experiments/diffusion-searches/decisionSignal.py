"""
Tests decision-point detection mechanisms for the branching text-diffusion
search (branching.py, oracle.py) that do NOT depend on a precomputed
spatial partition of the corpus (branching.py's own mechanism needs
corpus.HierarchicalClusterIndex, a fixed hierarchical k-means tree -- this
script is deliberately a REPLACEMENT candidate, not a modification of it).

Two mechanisms, both run on K independent FULL reverse-diffusion
completions (not the blurrier free x0Hat -- see diffusion.GaussianDiffusion.
reverseSteps' docstring on why x0Hat undershoots a true completion, ~0.8-0.94
cosine) sharing one text conditioning vector:

  1. Tag-histogram divergence. Each completion is mapped to its nearest
     REAL corpus font via a live, un-precomputed k-NN lookup
     (corpus.FontCorpus.nearest -- brute-force torch.cdist against the
     whole embeddings/all.json matrix; this is NOT a tree/grid/hash
     structure, just the corpus matrix itself, per the task's own
     framing). That font's DOMINANT tag (argmax of its 2280-d probability
     vector in embeddings/tags-latest-ViT_tags.json) is read off. A
     "decision point" fires when 2+ distinct dominant tags each collect
     at least minSupport completions.
  2. Statistical multimodality directly on the completions' whitened
     embeddings: GMM BIC(k=1 vs k=2), Hartigan's dip test (on the batch's
     own top principal component), and DBSCAN -- calibrated against a
     null baseline, since a previous live-clustering attempt in this
     project (see branching.py's docstring) measured indistinguishable
     from noise without that calibration.

Null baseline, two variants, both applied to the SAME real queries so
comparisons are paired:
  - "scrambled": the real query's own bge-large embedding with its 1024
    components permuted by an independent random permutation. Preserves
    the real vector's exact marginal value distribution and norm, destroys
    which value lands on which of the model's learned input weights.
  - "gaussian": a fresh random unit vector, decorrelated from any real
    query.

Usage:
    python3 experiments/diffusion-searches/decisionSignal.py --maxQueries 50 --numCompletions 30
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy import stats as scipystats

from corpus import FontCorpus
from dataset import PCAWhitener, EMBEDDINGS_PATH
from diffusion import DiffusionMLP, GaussianDiffusion

try:
    import diptest as _diptest
    HAS_DIPTEST = True
except ImportError:
    HAS_DIPTEST = False

CHECKPOINT_DIR = os.path.join("checkpoints", "diffusion")
TAGS_PATH = os.path.join("embeddings", "tags-latest-ViT_tags.json")
VOCAB_PATH = os.path.join("checkpoints", "retrieval", "latest", "ViT tags", "vocab.json")


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxQueries", type=int, default=50,
                         help="One query per distinct held-out font, matching evaluate_branching.py's convention.")
    parser.add_argument("--numCompletions", type=int, default=30,
                         help="Independent full reverse-diffusion completions per query (K).")
    parser.add_argument("--numSteps", type=int, default=50, help="Respaced reverse-diffusion steps.")
    parser.add_argument("--minSupport", type=int, default=3,
                         help="Minimum completions a dominant tag (or GMM/DBSCAN cluster) needs to count as a "
                              "real, non-noise group.")
    parser.add_argument("--localPCADim", type=int, default=5,
                         help="Dims of the batch's own local PCA used for GMM/DBSCAN (per-batch SVD, not a "
                              "precomputed structure -- refit fresh for every single query/null draw).")
    parser.add_argument("--checkpointDir", default=CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resultsOut", default=None)
    return parser.parse_args()


def loadTagsAndVocab():
    with open(TAGS_PATH, "r", encoding="utf-8") as f:
        tags = json.load(f)
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return tags, vocab


def dominantTagIds(names, tags, vocab):
    """names: list[str] (nearest real font per completion). Returns np.array of argmax tag indices."""
    ids = np.empty(len(names), dtype=int)
    for i, name in enumerate(names):
        vec = tags.get(name)
        ids[i] = int(np.argmax(vec)) if vec is not None else -1
    return ids


def tagHistogramTest(tagIds, minSupport):
    """
    Returns (numQualifyingGroups, topCounts sorted desc, concentration =
    top1Count / total). numQualifyingGroups>=2 is a "decision point";
    ==1 is a clear unanimous favorite; ==0 means nothing even reaches
    minSupport (small/fragmented batch).
    """
    valid = tagIds[tagIds >= 0]
    if valid.size == 0:
        return 0, [], 0.0
    counts = np.bincount(valid)
    order = np.argsort(-counts)
    topCounts = [int(counts[i]) for i in order if counts[i] > 0]
    qualifying = [c for c in topCounts if c >= minSupport]
    concentration = topCounts[0] / valid.size if topCounts else 0.0
    return len(qualifying), topCounts, concentration


def localPCA(x, dim):
    """x: [K, D] numpy. Centers and projects onto its own top `dim` principal components (per-batch SVD)."""
    centered = x - x.mean(axis=0, keepdims=True)
    dim = min(dim, x.shape[0] - 1, x.shape[1])
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:dim].T, s


def gmmTest(projected, minSupport):
    from sklearn.mixture import GaussianMixture
    k = projected.shape[0]
    covType = "diag" if projected.shape[1] > 1 else "spherical"
    gmm1 = GaussianMixture(n_components=1, covariance_type=covType, random_state=0, reg_covar=1e-3).fit(projected)
    bic1 = gmm1.bic(projected)
    best2, bic2 = None, np.inf
    for seed in range(3):
        try:
            gmm2 = GaussianMixture(n_components=2, covariance_type=covType, random_state=seed,
                                    reg_covar=1e-3).fit(projected)
        except Exception:
            continue
        b = gmm2.bic(projected)
        if b < bic2:
            bic2, best2 = b, gmm2
    if best2 is None:
        return {"deltaBIC": 0.0, "qualifyingClusters": 1 if k >= minSupport else 0}
    labels = best2.predict(projected)
    counts = np.bincount(labels, minlength=2)
    qualifying = int((counts >= minSupport).sum())
    return {"deltaBIC": float(bic1 - bic2), "qualifyingClusters": qualifying}


def dipTest(projected):
    """projected: [K, dim]; uses the top (already computed) local PC, column 0."""
    values = np.ascontiguousarray(projected[:, 0], dtype=np.float64)
    if HAS_DIPTEST:
        dip, pval = _diptest.diptest(values)
        return float(dip), float(pval)
    # Fallback: Sarle's bimodality coefficient (>5/9 suggestive of bimodality).
    skew = scipystats.skew(values)
    kurt = scipystats.kurtosis(values, fisher=False)
    n = values.size
    correction = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)) if n > 3 else 1.0
    bc = (skew ** 2 + 1) / (kurt + 3 * correction)
    return float(bc), float("nan")


def dbscanTest(projected, minSupport):
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    k = projected.shape[0]
    minPts = max(3, minSupport)
    nn = NearestNeighbors(n_neighbors=min(minPts, k - 1)).fit(projected)
    dists, _ = nn.kneighbors(projected)
    eps = float(np.median(dists[:, -1])) * 1.2 + 1e-6
    labels = DBSCAN(eps=eps, min_samples=minPts).fit_predict(projected)
    real = labels[labels >= 0]
    if real.size == 0:
        return {"numClusters": 0, "eps": eps}
    counts = np.bincount(real)
    qualifying = int((counts >= minSupport).sum())
    return {"numClusters": qualifying, "eps": eps}


def analyzeBatch(completions, corpus, tags, vocab, args):
    """completions: [K, pcaDim] torch, in corpus's whitened space."""
    _, nameRows = corpus.nearest(completions, k=1)
    names = [row[0] for row in nameRows]
    tagIds = dominantTagIds(names, tags, vocab)
    numTagGroups, topTagCounts, tagConcentration = tagHistogramTest(tagIds, args.minSupport)
    tagNames = [vocab[i] if i >= 0 else None for i in tagIds]

    x = completions.detach().cpu().numpy().astype(np.float64)
    projected, singularValues = localPCA(x, args.localPCADim)
    gmm = gmmTest(projected, args.minSupport)
    dip, dipP = dipTest(projected)
    dbscan = dbscanTest(projected, args.minSupport)

    return {
        "names": names,
        "tagIds": tagIds.tolist(),
        "tagNames": tagNames,
        "numTagGroups": numTagGroups,
        "topTagCounts": topTagCounts,
        "tagConcentration": tagConcentration,
        "numDistinctFonts": len(set(names)),
        "gmmDeltaBIC": gmm["deltaBIC"],
        "gmmQualifyingClusters": gmm["qualifyingClusters"],
        "dipStat": dip,
        "dipPval": dipP,
        "dbscanClusters": dbscan["numClusters"],
    }


def summarizeGroupTagMeaning(result, tags, vocab, corpus, topN=2):
    """
    For a query where the tag test fired (numTagGroups>=2), reports the
    actual tag names of the top groups and how far apart their member
    fonts sit in the whitened embedding space (mean pairwise distance
    between the two groups' completions' nearest real fonts) -- a concrete
    answer to "are the detected groups meaningfully different."
    """
    counts = {}
    for tagId in result["tagIds"]:
        if tagId >= 0:
            counts[tagId] = counts.get(tagId, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:topN]
    groups = []
    for tagId, count in ranked:
        memberFonts = [n for n, t in zip(result["names"], result["tagIds"]) if t == tagId]
        idx = [corpus.nameToIndex[n] for n in memberFonts if n in corpus.nameToIndex]
        groups.append({"tag": vocab[tagId], "count": count, "exampleFonts": memberFonts[:3], "memberIdx": idx})
    interGroupDist = None
    if len(groups) >= 2 and groups[0]["memberIdx"] and groups[1]["memberIdx"]:
        a = corpus.whitenedMatrix[groups[0]["memberIdx"]]
        b = corpus.whitenedMatrix[groups[1]["memberIdx"]]
        interGroupDist = torch.cdist(a, b).mean().item()
        intraA = torch.cdist(a, a).mean().item() if a.shape[0] > 1 else 0.0
        intraB = torch.cdist(b, b).mean().item() if b.shape[0] > 1 else 0.0
    else:
        intraA = intraB = None
    return groups, interGroupDist, intraA, intraB


def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(args.checkpointDir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.checkpointDir, "test_pairs.json")) as f:
        testPairs = json.load(f)

    whitener = PCAWhitener.load(os.path.join(args.checkpointDir, "whitener.npz"))
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device=device)
    tags, vocab = loadTagsAndVocab()

    import pickle
    with open(cachePathFor(config["sentenceModel"]), "rb") as f:
        sentenceCache = pickle.load(f)

    testPairs = [p for p in testPairs if p["font"] in corpus.nameToIndex and p["font"] in sentenceCache]
    byFont = {}
    for p in testPairs:
        byFont.setdefault(p["font"], p)
    queries = list(byFont.values())
    npRng = np.random.RandomState(args.seed)
    npRng.shuffle(queries)
    queries = queries[:args.maxQueries]
    print(f"{len(queries)} queries, one per distinct held-out font, K={args.numCompletions} completions each")

    model = DiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"],
                          hiddenDim=config["hiddenDim"], depth=config["depth"],
                          conditioning=config.get("conditioning", "concat")).to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpointDir, "checkpoint.pt"), map_location=device))
    model.eval()
    diffusion = GaussianDiffusion(timesteps=config["timesteps"], device=device)

    conditions = ["real", "scrambled", "gaussian"]
    allResults = {c: [] for c in conditions}
    exampleGroups = []

    for qi, pair in enumerate(queries):
        realText = torch.from_numpy(np.asarray(sentenceCache[pair["font"]][pair["index"]], dtype=np.float32))
        textDim = realText.shape[0]

        perm = rng.permutation(textDim)
        scrambledText = realText[perm].clone()

        gaussianText = torch.from_numpy(rng.normal(size=textDim).astype(np.float32))
        gaussianText = gaussianText / gaussianText.norm() * realText.norm()

        for cond, text in [("real", realText), ("scrambled", scrambledText), ("gaussian", gaussianText)]:
            torch.manual_seed(args.seed * 100003 + qi + hash(cond) % 997)
            textBatch = text.unsqueeze(0).expand(args.numCompletions, -1).contiguous().to(device)
            completions = diffusion.sample(model, textBatch, config["visualDim"], device=device,
                                            numSteps=args.numSteps)
            result = analyzeBatch(completions, corpus, tags, vocab, args)
            result["font"] = pair["font"]
            allResults[cond].append(result)

            if cond == "real" and result["numTagGroups"] >= 2 and len(exampleGroups) < 8:
                groups, interDist, intraA, intraB = summarizeGroupTagMeaning(result, tags, vocab, corpus)
                exampleGroups.append({"font": pair["font"], "groups": groups, "interGroupDist": interDist,
                                       "intraA": intraA, "intraB": intraB})

        print(f"\r{qi + 1}/{len(queries)} queries evaluated", end="")
    print()

    def summarize(cond):
        rs = allResults[cond]
        n = len(rs)
        return {
            "n": n,
            "tagGroupsFireRate": float(np.mean([r["numTagGroups"] >= 2 for r in rs])),
            "meanTagConcentration": float(np.mean([r["tagConcentration"] for r in rs])),
            "meanNumDistinctFonts": float(np.mean([r["numDistinctFonts"] for r in rs])),
            "gmmFireRate": float(np.mean([r["gmmQualifyingClusters"] >= 2 for r in rs])),
            "meanGmmDeltaBIC": float(np.mean([r["gmmDeltaBIC"] for r in rs])),
            "medianGmmDeltaBIC": float(np.median([r["gmmDeltaBIC"] for r in rs])),
            "meanDipStat": float(np.mean([r["dipStat"] for r in rs])),
            "dbscanFireRate": float(np.mean([r["dbscanClusters"] >= 2 for r in rs])),
            "meanDbscanClusters": float(np.mean([r["dbscanClusters"] for r in rs])),
        }

    print(f"\n{'metric':32s} {'real':>12s} {'scrambled':>12s} {'gaussian':>12s}")
    summaries = {c: summarize(c) for c in conditions}
    for key in summaries["real"]:
        if key == "n":
            continue
        print(f"{key:32s} " + " ".join(f"{summaries[c][key]:12.4f}" for c in conditions))

    # Paired significance tests (real vs each null), on the per-query metrics that matter most.
    print("\nPaired tests (real vs null), Wilcoxon signed-rank:")
    for nullCond in ["scrambled", "gaussian"]:
        realVals = np.array([r["gmmDeltaBIC"] for r in allResults["real"]])
        nullVals = np.array([r["gmmDeltaBIC"] for r in allResults[nullCond]])
        try:
            stat, p = scipystats.wilcoxon(realVals, nullVals)
        except ValueError:
            stat, p = float("nan"), float("nan")
        print(f"  gmmDeltaBIC   real vs {nullCond:10s}: W={stat:.2f} p={p:.4g} "
              f"(real mean {realVals.mean():.2f} vs null mean {nullVals.mean():.2f})")

        realTagFire = np.array([r["numTagGroups"] >= 2 for r in allResults["real"]], dtype=float)
        nullTagFire = np.array([r["numTagGroups"] >= 2 for r in allResults[nullCond]], dtype=float)
        try:
            stat, p = scipystats.wilcoxon(realTagFire, nullTagFire)
        except ValueError:
            stat, p = float("nan"), float("nan")
        print(f"  tagGroupFire  real vs {nullCond:10s}: W={stat:.2f} p={p:.4g} "
              f"(real rate {realTagFire.mean():.3f} vs null rate {nullTagFire.mean():.3f})")

        realDip = np.array([r["dipStat"] for r in allResults["real"]])
        nullDip = np.array([r["dipStat"] for r in allResults[nullCond]])
        try:
            stat, p = scipystats.wilcoxon(realDip, nullDip)
        except ValueError:
            stat, p = float("nan"), float("nan")
        print(f"  dipStat       real vs {nullCond:10s}: W={stat:.2f} p={p:.4g} "
              f"(real mean {realDip.mean():.4f} vs null mean {nullDip.mean():.4f})")

        realConc = np.array([r["tagConcentration"] for r in allResults["real"]])
        nullConc = np.array([r["tagConcentration"] for r in allResults[nullCond]])
        try:
            stat, p = scipystats.wilcoxon(realConc, nullConc)
        except ValueError:
            stat, p = float("nan"), float("nan")
        print(f"  tagConc       real vs {nullCond:10s}: W={stat:.2f} p={p:.4g} "
              f"(real mean {realConc.mean():.4f} vs null mean {nullConc.mean():.4f})")

        realDistinct = np.array([r["numDistinctFonts"] for r in allResults["real"]], dtype=float)
        nullDistinct = np.array([r["numDistinctFonts"] for r in allResults[nullCond]], dtype=float)
        try:
            stat, p = scipystats.wilcoxon(realDistinct, nullDistinct)
        except ValueError:
            stat, p = float("nan"), float("nan")
        print(f"  numDistinct   real vs {nullCond:10s}: W={stat:.2f} p={p:.4g} "
              f"(real mean {realDistinct.mean():.2f} vs null mean {nullDistinct.mean():.2f})")

    print(f"\nExample real-query tag-group splits (up to 8, only where numTagGroups>=2):")
    for ex in exampleGroups:
        groupStr = "; ".join(f"{g['tag']}(n={g['count']}, e.g. {g['exampleFonts']})" for g in ex["groups"])
        interStr = f"{ex['interGroupDist']:.3f}" if ex["interGroupDist"] is not None else "n/a"
        intraStr = (f"{ex['intraA']:.3f}/{ex['intraB']:.3f}" if ex["intraA"] is not None else "n/a")
        print(f"  target={ex['font']!r}: {groupStr} | inter-group whitened dist={interStr}, "
              f"intra-group dists={intraStr}")

    if args.resultsOut:
        with open(args.resultsOut, "w") as f:
            json.dump({"args": vars(args), "summaries": summaries,
                       "exampleGroups": [{"font": e["font"], "interGroupDist": e["interGroupDist"],
                                          "groups": [{"tag": g["tag"], "count": g["count"]} for g in e["groups"]]}
                                         for e in exampleGroups]}, f, indent=2)
        print(f"\nwrote summary to {args.resultsOut}")


if __name__ == "__main__":
    main()
