"""
Evaluates the branching/decision-point interactive search (branching.py,
oracle.py) over held-out test queries: runs a full oracle-driven session
per query at each of several oracle-error ("noise") probabilities, and
reports recall@k plus the mean number of decision points a session
needed at each level, so noiseProb=0 vs. noiseProb>0 shows directly how
much the whole pipeline's accuracy depends on the user picking correctly
at each branch. Also reports (unless --skipBaseline) a single-shot,
no-branching baseline at the same particle budget, as a reference for
whether branching+oracle-guidance helps, hurts, or roughly matches a
single unguided sample at noiseProb=0.

Decision points are navigated down a precomputed HIERARCHICAL corpus
partition (corpus.HierarchicalClusterIndex), capped at --maxRounds
(default 5) decisions -- see corpus.py and branching.py's docstrings for
why this replaced an earlier flat (single-level) partition: navigating
the same tree with an OMNISCIENT oracle (no diffusion model involved at
all) got a median final leaf of 4 fonts within 5 rounds on this exact
corpus, implied recall@10 ~97%, versus ~10% for the flat-partition
version -- the flat partition, not the embedding space, was the
bottleneck. Building the tree is a real one-time cost (~90s for the
defaults below on this machine); --treeCache saves/loads it so repeated
runs during iteration don't re-pay that cost.

Per-query randomness is paired across noise levels deliberately: both the
diffusion sampling RNG and the oracle's own RNG are reseeded from the
SAME per-query seed before every noise level's session, so the only thing
that differs between rows for a given query is which decisions the oracle
actually got to make noisily -- not incidental resampling differences.

Success uses an ACCEPTANCE SET, not the single exact font: a query
"succeeds" at k if ANY of the target's own top --acceptanceSize nearest
neighbors (in the corpus's whitened space) lands in the top k of the
final best-of-N ranking -- matching utils/search.py's GridFeedbackSearch
simulated-user convention (see CLAUDE.md), not evaluate.py's stricter
single-font metric. At this corpus's density, a great many fonts are
near-indistinguishable near-duplicates of each other, so "landed on a
near-identical font, not the one exact database row" is a legitimate
product success.

    python3 experiments/diffusion-searches/evaluate_branching.py --maxQueries 50
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import (PCAWhitener, TextCenterer, EMBEDDINGS_PATH, loadTagPresenceCache, concatenateTagPresence,
                      loadTfidfFeatureCache, concatenateTfidfCache)
from diffusion import DiffusionMLP, TwoPhaseDiffusionMLP, GaussianDiffusion
from oracle import runSession

CHECKPOINT_DIR = os.path.join("checkpoints", "diffusion")
K_VALUES = [1, 5, 10, 50, 100]


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxQueries", type=int, default=30,
                         help="One query per DISTINCT font (matching diagnose_*.py's convention), "
                              "sampled from the held-out test split.")
    parser.add_argument("--numParticles", type=int, default=40)
    parser.add_argument("--numSteps", type=int, default=50,
                         help="Respaced reverse-diffusion steps -- see diffusion.GaussianDiffusion.respacedSteps.")
    parser.add_argument("--noiseLevels", default="0.0,0.1,0.2,0.3,0.5",
                         help="Comma-separated oracle per-decision-point error probabilities to sweep.")
    parser.add_argument("--maxModes", type=int, default=4)
    parser.add_argument("--minClusterSize", type=int, default=2,
                         help="Minimum particles (out of --numParticles) a hierarchy child needs to count "
                              "as a real candidate branch. Lower is better here than intuition suggests: a "
                              "high bar (e.g. 8/40=20%%) means the model's TOP pick auto-descends whenever "
                              "no second child clears it, even though top-pick accuracy alone measured only "
                              "~47%% at the root -- a confident-but-wrong model then walks right past the "
                              "oracle with no chance to correct it. A low bar surfaces weaker secondary "
                              "candidates as real choices instead, so the (reliable, at low noiseProb) "
                              "oracle gets consulted far more often. Swept 1/2/3/4/8 at persistFor 3-5;  "
                              "2-3 measured best (recall@10 roughly tripled, recall@50/100 nearly doubled, "
                              "vs. 8) -- see README for the full sweep.")
    parser.add_argument("--persistFor", type=int, default=4,
                         help="A split must show the same qualifying child set for this many consecutive "
                              "checks before it counts as a real decision point -- see branching."
                              "findDecisionPoint's docstring for why a single instant isn't enough.")
    parser.add_argument("--maxRounds", type=int, default=5,
                         help="Hard cap on decision points per session.")
    parser.add_argument("--reserveSize", type=int, default=5,
                         help="Completions stashed from EACH rejected branch at every asked decision, "
                              "for the 'hedge' final-ranking method -- see oracle.runSession.")
    parser.add_argument("--hedgeMaxBranchSize", type=int, default=500,
                         help="A rejected branch only joins the hedge if its own membership is at most "
                              "this large -- large (early-round) rejections are trusted outright rather "
                              "than hedged on, since including thousands of fonts reintroduces the same "
                              "dilution a full-corpus hedge had. See oracle.runSession's docstring.")
    parser.add_argument("--forceAskThreshold", type=int, default=1000,
                         help="Nodes larger than this NEVER auto-descend, even with only one clearly "
                              "qualifying child -- the runner-up is padded in and the oracle is asked to "
                              "confirm/override, since an unsupervised wrong pick there is much costlier "
                              "than at a small node. See oracle.runSession's docstring.")
    parser.add_argument("--guidanceScale", type=float, default=1.0,
                         help="Classifier-free guidance scale (1.0 = disabled/original behavior). A "
                              "full-corpus coverage experiment found reachability is NOT the bottleneck "
                              "(~96%%+ of the corpus is approximated within 2x real-neighbor tightness by "
                              "SOME generated sample already) but per-query PRECISION is (a font's own "
                              "query lands near ITS OWN target only 7.45%% of the time within 2x); CFG "
                              "measured ~1.5x improvement on that at scale 2-4 on 3000 held-out queries. "
                              "See diffusion.GaussianDiffusion.reverseSteps' docstring.")
    parser.add_argument("--branchingFactor", type=int, default=10,
                         help="Children per internal node of the hierarchical corpus partition.")
    parser.add_argument("--treeDepth", type=int, default=5,
                         help="Max depth of the hierarchical corpus partition -- combined with "
                              "--branchingFactor this bounds addressable resolution at branchingFactor^"
                              "treeDepth leaves; also naturally caps rounds needed (kept in sync with "
                              "--maxRounds by default).")
    parser.add_argument("--minLeafSize", type=int, default=5)
    parser.add_argument("--treeCache", default=os.path.join(CHECKPOINT_DIR, "hierarchy.pkl"),
                         help="Path to save/load the built hierarchical partition -- building it is a real "
                              "one-time cost, this avoids re-paying it on every run. Delete the file (or "
                              "pass a new path) to force a rebuild after changing --branchingFactor/"
                              "--treeDepth/--minLeafSize.")
    parser.add_argument("--rebuildTree", action="store_true", help="Ignore --treeCache and rebuild.")
    parser.add_argument("--acceptanceSize", type=int, default=10,
                         help="A session succeeds at k if any of the target's own top-N nearest neighbors "
                              "(matching GridFeedbackSearch's simulated-user convention) lands in the top "
                              "k of the final ranking -- not just the single exact font. See module docstring.")
    parser.add_argument("--checkEvery", type=int, default=1,
                         help="Check for a decision point every this many respaced schedule steps -- cheap "
                              "now that checks are a nearest-fixed-centroid lookup, not live clustering.")
    parser.add_argument("--minScheduleFraction", type=float, default=0.1,
                         help="Skip decision-point checks during this leading fraction of the schedule, "
                              "where x_t is still noise-dominated regardless of conditioning.")
    parser.add_argument("--topPerCluster", type=int, default=10)
    parser.add_argument("--numPreviews", type=int, default=5,
                         help="Independent full-completion rollouts per cluster used to resolve preview fonts.")
    parser.add_argument("--skipBaseline", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resultsOut", default=None,
                         help="Optional path to dump per-query, per-noiseLevel, per-scoring-method raw "
                              "results as JSON, for later charting/analysis.")
    parser.add_argument("--checkpointDir", default=CHECKPOINT_DIR,
                         help="Diffusion checkpoint to evaluate -- point this at a variant trained with "
                              "train.py --checkpointDir (e.g. a --conditioning film run) to compare "
                              "against the default. --treeCache is independent of this (the corpus tree "
                              "doesn't depend on which diffusion checkpoint generates the particles).")
    return parser.parse_args()


def singleShotFinalX(diffusion, model, text, visualDim, numParticles, numSteps, device,
                      guidanceScale=1.0, nullText=None):
    textBatch = text.unsqueeze(0).expand(numParticles, -1).contiguous().to(device)
    nullTextBatch = (nullText.unsqueeze(0).expand(numParticles, -1).contiguous().to(device)
                      if nullText is not None else None)
    return diffusion.sample(model, textBatch, visualDim, device=device, numSteps=numSteps,
                             guidanceScale=guidanceScale, nullText=nullTextBatch)


def bestRank(finalX, corpus, acceptanceIdx):
    """
    Best-of-N ranking in the corpus's whitened space (ascending distance),
    generalized to an acceptance SET: returns the best (smallest,
    0-indexed) rank among acceptanceIdx, i.e. the number of candidate
    fonts that beat the closest acceptable match. Reduces to a single-
    font rank when acceptanceIdx has one element. This is the "hard"
    metric: finalX's particles only ever explored the branch the session
    hard-pruned down to, so this implicitly inherits that commitment even
    though it nominally ranks against the WHOLE corpus.
    """
    finalX = finalX.detach().to(corpus.device)
    dists = torch.cdist(finalX, corpus.whitenedMatrix)  # [numParticles, N]
    bestDists, _ = dists.min(dim=0)  # [N] best-of-particles per candidate font
    bestAcceptanceDist = bestDists[acceptanceIdx].min()
    return int((bestDists < bestAcceptanceDist).sum().item())


def hedgeRank(session, corpus, acceptanceIdx):
    """
    Pools the main path's finalX with "reserveX" (rejected-branch
    completions, see oracle.runSession), but -- unlike a naive pooled
    bestRank -- ranks ONLY within "candidatePool" (every branch actually
    considered this session: the final path plus every rejected branch's
    real membership), not the full 39,421-font corpus. Pooling against
    the full corpus was tried first and measured WORSE than finalX alone:
    more particles help distractor candidates just as readily as the true
    one, and there are vastly more distractors than true positives, so it
    diluted rather than helped. Restricting to what was actually shown
    avoids that dilution while still giving a rejected-but-maybe-correct
    branch a chance to surface the target.

    If the acceptance set isn't even in candidatePool (the target's own
    branch was never considered at all, at any round), returns
    len(candidatePool) -- i.e. "worse than everything considered," which
    is honest: no amount of re-ranking within the wrong set can find it.
    """
    candidateIdx = sorted(session["candidatePool"])
    pooled = torch.cat([session["finalX"], session["reserveX"]], dim=0).detach().to(corpus.device)
    dists = torch.cdist(pooled, corpus.whitenedMatrix[candidateIdx])  # [numPooled, len(candidateIdx)]
    bestDists, _ = dists.min(dim=0)  # [len(candidateIdx)]

    posMap = {idx: i for i, idx in enumerate(candidateIdx)}
    acceptancePositions = [posMap[i] for i in acceptanceIdx if i in posMap]
    if not acceptancePositions:
        return len(candidateIdx)

    bestAcceptanceDist = bestDists[acceptancePositions].min()
    return int((bestDists < bestAcceptanceDist).sum().item())


def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.checkpointDir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.checkpointDir, "test_pairs.json")) as f:
        testPairs = json.load(f)

    whitener = PCAWhitener.load(os.path.join(args.checkpointDir, "whitener.npz"))
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device=device)
    with open(cachePathFor(config["sentenceModel"]), "rb") as f:
        sentenceCache = pickle.load(f)
    if config.get("centerText"):
        centerer = TextCenterer.load(os.path.join(args.checkpointDir, "textCenter.npz"))
        sentenceCache = centerer.applyToCache(sentenceCache)
    if config.get("tagConditioning"):
        tagCache = loadTagPresenceCache()
        sentenceCache = concatenateTagPresence(sentenceCache, tagCache)
    if config.get("tfidfDim", 0) > 0 and config.get("tfidfOnly", False):
        sentenceCache = loadTfidfFeatureCache(config["tfidfDim"])
    elif config.get("tfidfDim", 0) > 0:
        tfidfCache = loadTfidfFeatureCache(config["tfidfDim"])
        sentenceCache = concatenateTfidfCache(sentenceCache, tfidfCache)

    nullText = None
    if args.guidanceScale != 1.0:
        allVectors = np.concatenate(list(sentenceCache.values()), axis=0)
        nullText = torch.from_numpy(allVectors.mean(axis=0).astype(np.float32)).to(device)
        print(f"classifier-free guidance enabled: scale={args.guidanceScale}, "
              f"nullText=corpus-wide mean query embedding")

    testPairs = [p for p in testPairs if p["font"] in corpus.nameToIndex]
    byFont = {}
    for p in testPairs:
        byFont.setdefault(p["font"], p)
    queries = list(byFont.values())
    rng = np.random.RandomState(args.seed)
    rng.shuffle(queries)
    if args.maxQueries is not None:
        queries = queries[:args.maxQueries]
    print(f"{len(queries)} queries, one per distinct held-out font")

    if config.get("architecture", "standard") == "shapeI":
        model = TwoPhaseDiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"],
                                      hiddenDim=config["hiddenDim"], timeDim=config.get("timeDim", 64),
                                      numPlainBlocks=config.get("numPlainBlocks", 1),
                                      numConditionedBlocks=config.get("numConditionedBlocks", 2)).to(device)
    else:
        model = DiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"],
                              hiddenDim=config["hiddenDim"], depth=config["depth"],
                              conditioning=config.get("conditioning", "concat")).to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpointDir, "checkpoint.pt"), map_location=device))
    model.eval()
    diffusion = GaussianDiffusion(timesteps=config["timesteps"], device=device)

    if not args.rebuildTree and os.path.exists(args.treeCache):
        print(f"loading cached hierarchy from {args.treeCache}")
        hierarchicalIndex = HierarchicalClusterIndex.load(args.treeCache)
    else:
        print(f"building hierarchy (branchingFactor={args.branchingFactor}, depth={args.treeDepth}, "
              f"minLeafSize={args.minLeafSize}) -- this can take ~a minute...")
        hierarchicalIndex = HierarchicalClusterIndex.fit(
            corpus, branchingFactor=args.branchingFactor, maxDepth=args.treeDepth, minLeafSize=args.minLeafSize)
        os.makedirs(os.path.dirname(args.treeCache), exist_ok=True)
        hierarchicalIndex.save(args.treeCache)
        print(f"saved hierarchy to {args.treeCache}")

    noiseLevels = [float(x) for x in args.noiseLevels.split(",")]
    methods = ["hard", "hedge"]
    results = {level: {m: {"hits": {k: 0 for k in K_VALUES}, "ranks": []} for m in methods}
               for level in noiseLevels}
    for level in noiseLevels:
        results[level]["rounds"] = []
        results[level]["leafSizes"] = []
        results[level]["leafHasTarget"] = []
        results[level]["leafHasAcceptable"] = []
        results[level]["candidatePoolSizes"] = []
    baselineHits = {k: 0 for k in K_VALUES}

    for qi, pair in enumerate(queries):
        text = torch.from_numpy(np.asarray(sentenceCache[pair["font"]][pair["index"]], dtype=np.float32)).to(device)
        trueIndex = corpus.nameToIndex[pair["font"]]
        _, acceptanceRows = corpus.nearest(corpus.whitenedMatrix[trueIndex].unsqueeze(0), k=args.acceptanceSize)
        acceptanceIdx = [corpus.nameToIndex[n] for n in acceptanceRows[0]]
        querySeed = args.seed * 100003 + qi

        if not args.skipBaseline:
            torch.manual_seed(querySeed)
            finalX = singleShotFinalX(diffusion, model, text, config["visualDim"], args.numParticles,
                                       args.numSteps, device, guidanceScale=args.guidanceScale, nullText=nullText)
            rank = bestRank(finalX, corpus, acceptanceIdx)
            for k in K_VALUES:
                if rank < k:
                    baselineHits[k] += 1

        for level in noiseLevels:
            torch.manual_seed(querySeed)
            oracleRng = np.random.default_rng(querySeed)
            session = runSession(diffusion, model, text, acceptanceIdx, corpus, hierarchicalIndex,
                                  config["visualDim"], numParticles=args.numParticles, numSteps=args.numSteps,
                                  maxModes=args.maxModes, topPerCluster=args.topPerCluster,
                                  numPreviews=args.numPreviews, checkEvery=args.checkEvery,
                                  minScheduleFraction=args.minScheduleFraction,
                                  minClusterSize=args.minClusterSize, persistFor=args.persistFor,
                                  maxRounds=args.maxRounds, reserveSize=args.reserveSize,
                                  hedgeMaxBranchSize=args.hedgeMaxBranchSize,
                                  forceAskThreshold=args.forceAskThreshold,
                                  guidanceScale=args.guidanceScale, nullText=nullText,
                                  noiseProb=level, rng=oracleRng, device=device)

            ranks = {
                "hard": bestRank(session["finalX"], corpus, acceptanceIdx),
                "hedge": hedgeRank(session, corpus, acceptanceIdx),
            }
            for m in methods:
                results[level][m]["ranks"].append(ranks[m])
                for k in K_VALUES:
                    if ranks[m] < k:
                        results[level][m]["hits"][k] += 1

            results[level]["rounds"].append(session["rounds"])
            finalMembers = session["finalNode"].memberIndices
            results[level]["leafSizes"].append(len(finalMembers))
            results[level]["leafHasTarget"].append(trueIndex in finalMembers)
            results[level]["leafHasAcceptable"].append(any(i in finalMembers for i in acceptanceIdx))
            results[level]["candidatePoolSizes"].append(len(session["candidatePool"]))

        print(f"\r{qi + 1}/{len(queries)} queries evaluated", end="")
    print()

    total = len(queries)
    actualSteps = min(args.numSteps, config["timesteps"])

    if not args.skipBaseline:
        print(f"\nSingle-shot baseline (no branching, {args.numParticles} samples/query, "
              f"{actualSteps}/{config['timesteps']} steps), {total} queries:")
        for k in K_VALUES:
            print(f"  recall@{k}: {baselineHits[k] / total:.4f}")

    print(f"\nBranching + oracle search, {total} queries, {args.numParticles} particles/round, "
          f"{actualSteps}/{config['timesteps']} steps, branchingFactor={args.branchingFactor}, "
          f"maxRounds={args.maxRounds}, reserveSize={args.reserveSize}. Two final-scoring methods compared:\n"
          f"  hard  = rank (against the FULL corpus) by the hard-pruned tree path's own final particles only\n"
          f"  hedge = rank ONLY within candidatePool (every branch actually considered, typically much "
          f"smaller than the full corpus -- see sizes below) using finalX + reserveX pooled")
    for level in noiseLevels:
        rounds = np.array(results[level]["rounds"])
        leafSizes = np.array(results[level]["leafSizes"])
        poolSizes = np.array(results[level]["candidatePoolSizes"])
        branched = rounds > 0
        leafHasTarget = np.array(results[level]["leafHasTarget"])
        leafHasAcceptable = np.array(results[level]["leafHasAcceptable"])
        print(f"\n  noiseProb={level}:")
        print(f"    mean decision points/session: {rounds.mean():.2f}  (max {rounds.max()}), "
              f"{branched.mean() * 100:.0f}% of sessions needed at least one decision")
        print(f"    final node size: mean {leafSizes.mean():.1f}  median {np.median(leafSizes):.0f}")
        print(f"    candidatePool size (hedge's search space): mean {poolSizes.mean():.1f}  "
              f"median {np.median(poolSizes):.0f}")
        print(f"    final group actually contains the exact target font: {leafHasTarget.mean():.4f}")
        print(f"    final group actually contains an acceptance-set font: {leafHasAcceptable.mean():.4f}")
        for m in methods:
            hits = results[level][m]["hits"]
            print(f"    [{m}]  " + "  ".join(f"recall@{k}={hits[k] / total:.4f}" for k in K_VALUES))

    if args.resultsOut:
        dumpable = {
            "args": vars(args),
            "total": total,
            "baseline": {k: baselineHits[k] / total for k in K_VALUES},
            "byNoiseLevel": {
                str(level): {
                    "meanRounds": float(np.mean(results[level]["rounds"])),
                    "leafHasTarget": float(np.mean(results[level]["leafHasTarget"])),
                    "leafHasAcceptable": float(np.mean(results[level]["leafHasAcceptable"])),
                    "methods": {
                        m: {f"recall@{k}": results[level][m]["hits"][k] / total for k in K_VALUES}
                        for m in methods
                    },
                }
                for level in noiseLevels
            },
        }
        with open(args.resultsOut, "w") as f:
            json.dump(dumpable, f, indent=2)
        print(f"\nwrote raw results to {args.resultsOut}")


if __name__ == "__main__":
    main()
