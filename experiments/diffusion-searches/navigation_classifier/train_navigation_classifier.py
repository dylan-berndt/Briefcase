"""
NEW hypothesis, not tried elsewhere this session: every tree-navigation
decision so far has been made indirectly -- run full diffusion sampling,
then nearest-centroid-assign the result. That's using a slow, noisy
generative process as a stand-in for what is actually a direct
classification problem: given (text, current node), which child contains
the target? Train a small, dedicated classifier for exactly that,
supervised (cross-entropy against the KNOWN true child along the target's
real tree path) -- no diffusion sampling, no RL, no reward sparsity, no
GRPO instability. Ground-truth (text, node, true-child) examples are
essentially free to generate (pure tree lookup, no model calls at all),
so this is a very cheap experiment to run before committing more time to
GRPO variants.

Scores each child individually (bilinear-ish: MLP over
[text; nodeCentroid; childCentroid]) rather than a fixed-size softmax
head, so one network handles nodes with any number of children.

    python3 experiments/diffusion-searches/navigation_classifier/train_navigation_classifier.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_baseDir = _os.path.dirname(_scriptDir)
for _p in [_baseDir] + [_os.path.join(_baseDir, _d) for _d in _os.listdir(_baseDir)
                         if _os.path.isdir(_os.path.join(_baseDir, _d))]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse

from runConfig import loadRunConfig
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from corpus import FontCorpus, HierarchicalClusterIndex, nodeVector
from dataset import PCAWhitener, loadRaw, splitQueryCache, EMBEDDINGS_PATH, \
    loadTfidfFeatureCache, concatenateTfidfCache

MAX_CHILDREN = 16


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def trueChildAt(node, targetIdx):
    for i, child in enumerate(node.children):
        if targetIdx in child.memberIndices:
            return i, child
    return None, None


def buildExamples(pairs, sentenceCache, corpus, hier, maxDepth, maxExamples, seed):
    """
    Stores a NODE REFERENCE per example (plus the raw text vector/trueIdx/
    depth), not precomputed node/child vectors -- NavDataset materializes
    those lazily in __getitem__ instead. Many examples share the same tree
    node (every query passing through the root, say), so eagerly storing a
    full padded [MAX_CHILDREN, nodeDim] array per example duplicates that
    node's data across every example that touches it; with nodeDim widened
    by --nodeTextWeight (64 -> 64+textDim, e.g. 1088 for BGE) that
    duplication is large enough to exhaust memory outright (measured: killed
    by the OS at maxTrainExamples=300000). Lazy materialization trades a
    small, cheap per-epoch recompute (plain numpy concatenation) for O(1)
    storage per example instead of O(nodeDim).
    """
    rng = np.random.RandomState(seed)
    order = np.arange(len(pairs))
    rng.shuffle(order)
    examples = []
    depthCounts = {}
    for idx in order:
        name, qi = pairs[idx]
        if name not in corpus.nameToIndex:
            continue
        targetIdx = corpus.nameToIndex[name]
        textVec = np.asarray(sentenceCache[name][qi], dtype=np.float32)
        node = hier.root
        depth = 0
        while node.children and depth < maxDepth:
            trueIdx, trueChild = trueChildAt(node, targetIdx)
            if trueIdx is None:
                break
            examples.append((textVec, node, trueIdx, depth))
            depthCounts[depth] = depthCounts.get(depth, 0) + 1
            node = trueChild
            depth += 1
        if len(examples) >= maxExamples:
            break
    return examples, depthCounts


class NavDataset(Dataset):
    def __init__(self, examples, nodeTextWeight=0.0, nodeTextDim=0):
        self.examples = examples
        self.nodeTextWeight = nodeTextWeight
        self.nodeTextDim = nodeTextDim

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        textVec, node, trueIdx, depth = self.examples[i]
        childCentroids = np.stack([nodeVector(c, self.nodeTextWeight, self.nodeTextDim) for c in node.children])
        numChildren = childCentroids.shape[0]
        padded = np.zeros((MAX_CHILDREN, childCentroids.shape[1]), dtype=np.float32)
        padded[:numChildren] = childCentroids
        mask = np.zeros(MAX_CHILDREN, dtype=bool)
        mask[:numChildren] = True
        return {
            "text": torch.from_numpy(textVec),
            "node": torch.from_numpy(nodeVector(node, self.nodeTextWeight, self.nodeTextDim)),
            "children": torch.from_numpy(padded),
            "mask": torch.from_numpy(mask),
            "trueIdx": trueIdx,
            "depth": depth,
        }

    @staticmethod
    def collate(samples):
        return {
            "text": torch.stack([s["text"] for s in samples]),
            "node": torch.stack([s["node"] for s in samples]),
            "children": torch.stack([s["children"] for s in samples]),
            "mask": torch.stack([s["mask"] for s in samples]),
            "trueIdx": torch.tensor([s["trueIdx"] for s in samples], dtype=torch.long),
            "depth": torch.tensor([s["depth"] for s in samples], dtype=torch.long),
        }


class NavigationClassifier(nn.Module):
    def __init__(self, textDim, pcaDim, hiddenDim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(textDim + pcaDim * 2, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, 1),
        )

    def forward(self, text, node, children, mask):
        B, maxChildren, pcaDim = children.shape
        textExp = text.unsqueeze(1).expand(-1, maxChildren, -1)
        nodeExp = node.unsqueeze(1).expand(-1, maxChildren, -1)
        inp = torch.cat([textExp, nodeExp, children], dim=-1)
        scores = self.net(inp).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores


def parseArgs():
    """Run settings are in configs/navigationClassifier.json (copy it per variant). Keys beyond the obvious:
      baseCheckpoint  only used for whitener.npz / config.json (pcaDim, textDim, testFraction)
      depthWeights    per-depth loss weights, a list [1, 3, 1, 1, 1] or the string "1,3,1,1,1"; null = uniform
      textMode        "bge" dense sentence embedding; "lsa" precomputed TF-IDF/LSA features instead
                      (build_tfidf_cache.py); "concat" both. tfidfDim is the cached LSA width
      nodeTextWeight  >0 appends weight * node.textCentroid to every node/child vector (needs a tree built with
                      build_text_tree.py's auxMatrix); 0 reproduces prior behavior on any tree
    The defaults are the script's original ones (v1 scale); the reported-best v4 used 771k examples / 60 epochs /
    hiddenDim 384."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join("configs", "navigationClassifier.json"))
    parser.add_argument("--checkpointDir", default=None, help="override the config's checkpointDir")
    args = parser.parse_args()
    checkpointDir = args.checkpointDir
    del args.checkpointDir
    cfg = loadRunConfig(args.config, args)
    if checkpointDir:
        cfg.checkpointDir = checkpointDir
    return cfg


def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    with open(os.path.join(args.baseCheckpoint, "config.json")) as f:
        config = json.load(f)

    fontEmbeddings, sentenceCache = loadRaw(sentenceCachePath=cachePathFor(args.sentenceModel))

    if args.textMode == "lsa":
        sentenceCache = loadTfidfFeatureCache(args.tfidfDim)
        textDim = args.tfidfDim
    elif args.textMode == "concat":
        tfidfCache = loadTfidfFeatureCache(args.tfidfDim)
        sentenceCache = concatenateTfidfCache(sentenceCache, tfidfCache)
        textDim = config["textDim"] + args.tfidfDim
    else:
        textDim = config["textDim"]

    # names must come from whichever cache is ACTUALLY in use post text-mode switch --
    # splitQueryCache does a direct dict lookup with no fallback, and the LSA cache's
    # key set isn't guaranteed identical to the BGE cache's.
    names = sorted(set(fontEmbeddings.keys()) & set(sentenceCache.keys()))

    trainCache, testCache = splitQueryCache(sentenceCache, names, testFraction=config["testFraction"], seed=config["seed"])
    whitener = PCAWhitener.load(os.path.join(args.baseCheckpoint, "whitener.npz"))
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device="cpu")
    hier = HierarchicalClusterIndex.load(args.treeCache)

    nodeTextDim = 0
    if args.nodeTextWeight > 0:
        if hier.root.textCentroid is None:
            raise ValueError(f"--nodeTextWeight={args.nodeTextWeight} but {args.treeCache}'s root has no "
                              f"textCentroid -- rebuild the tree with build_text_tree.py's auxMatrix first.")
        nodeTextDim = hier.root.textCentroid.shape[0]
    nodeDim = config["visualDim"] + nodeTextDim

    trainPairs = [(name, i) for name, vectors in trainCache.items() for i in range(len(vectors))]
    testPairs = [(name, i) for name, vectors in testCache.items() for i in range(len(vectors))]
    print(f"{len(trainPairs)} train pairs, {len(testPairs)} test pairs available")

    print("building training examples (pure tree lookup, no model calls) ...")
    trainExamples, trainDepthCounts = buildExamples(trainPairs, trainCache, corpus, hier, args.maxDepth,
                                                       args.maxTrainExamples, args.seed)
    testExamples, testDepthCounts = buildExamples(testPairs, testCache, corpus, hier, args.maxDepth,
                                                     args.maxTestExamples, args.seed + 1)
    print(f"{len(trainExamples)} train examples (by depth: {trainDepthCounts}), "
          f"{len(testExamples)} test examples (by depth: {testDepthCounts})")

    trainLoader = DataLoader(NavDataset(trainExamples, args.nodeTextWeight, nodeTextDim), batch_size=args.batchSize,
                              shuffle=True, collate_fn=NavDataset.collate)
    testLoader = DataLoader(NavDataset(testExamples, args.nodeTextWeight, nodeTextDim), batch_size=args.batchSize,
                             shuffle=False, collate_fn=NavDataset.collate)

    model = NavigationClassifier(textDim, nodeDim, args.hiddenDim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learningRate)

    depthWeights = None
    if args.depthWeights:
        weights = args.depthWeights.split(",") if isinstance(args.depthWeights, str) else args.depthWeights
        depthWeights = torch.tensor([float(w) for w in weights], device=device)

    def evaluate(loader):
        model.eval()
        top1, top3, total = 0, 0, 0
        byDepth = {}
        with torch.no_grad():
            for batch in loader:
                text = batch["text"].to(device)
                node = batch["node"].to(device)
                children = batch["children"].to(device)
                mask = batch["mask"].to(device)
                trueIdx = batch["trueIdx"].to(device)
                depth = batch["depth"]

                scores = model(text, node, children, mask)
                top3Idx = scores.topk(3, dim=1).indices
                pred1 = scores.argmax(dim=1)
                correct1 = (pred1 == trueIdx)
                correct3 = (top3Idx == trueIdx.unsqueeze(1)).any(dim=1)

                top1 += correct1.sum().item()
                top3 += correct3.sum().item()
                total += trueIdx.shape[0]

                for d in depth.unique().tolist():
                    m = (depth == d)
                    byDepth.setdefault(d, [0, 0, 0])
                    byDepth[d][0] += correct1[m.to(device)].sum().item()
                    byDepth[d][1] += correct3[m.to(device)].sum().item()
                    byDepth[d][2] += m.sum().item()
        return top1 / total, top3 / total, byDepth

    for epoch in range(args.epochs):
        model.train()
        totalLoss, steps = 0.0, 0
        for batch in trainLoader:
            text = batch["text"].to(device)
            node = batch["node"].to(device)
            children = batch["children"].to(device)
            mask = batch["mask"].to(device)
            trueIdx = batch["trueIdx"].to(device)
            depth = batch["depth"].to(device)

            scores = model(text, node, children, mask)
            if depthWeights is not None:
                perExample = nn.functional.cross_entropy(scores, trueIdx, reduction="none")
                loss = (perExample * depthWeights[depth]).mean()
            else:
                loss = nn.functional.cross_entropy(scores, trueIdx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            steps += 1

        if (epoch + 1) % 3 == 0 or epoch == args.epochs - 1:
            top1, top3, byDepth = evaluate(testLoader)
            print(f"epoch {epoch + 1}/{args.epochs}  trainLoss={totalLoss / steps:.4f}  "
                  f"testTop1={top1:.3f}  testTop3={top3:.3f}")

    top1, top3, byDepth = evaluate(testLoader)
    print(f"\nFINAL pooled: top1={top1:.4f}  top3={top3:.4f}")
    print("by depth:")
    for d in sorted(byDepth):
        c1, c3, n = byDepth[d]
        print(f"  depth {d}: n={n}  top1={c1/n:.3f}  top3={c3/n:.3f}")

    print("\nFOR COMPARISON (diffusion+nearest-centroid, from tree_dip_investigation.py, same tree):")
    print("  depth 0: top1=0.433-0.440  top3=0.753-0.780")
    print("  depth 1: top1=0.300-0.413  top3=0.653-0.787")
    print("  depth 2: top1=0.220-0.353  top3=0.600-0.713")
    print("  depth 3: top1=0.193-0.287  top3=0.467-0.793")
    print("  depth 4: top1=0.331-0.368  top3=0.705-0.735")
    print("  POOLED:  top1=0.324-0.342  top3=0.671-0.724")

    os.makedirs(args.checkpointDir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.checkpointDir, "checkpoint.pt"))
    with open(os.path.join(args.checkpointDir, "config.json"), "w") as f:
        json.dump({"textDim": textDim, "pcaDim": config["visualDim"], "hiddenDim": args.hiddenDim,
                    "maxChildren": MAX_CHILDREN, "treeCache": args.treeCache,
                    "baseCheckpoint": args.baseCheckpoint, "sentenceModel": args.sentenceModel,
                    "textMode": args.textMode, "tfidfDim": args.tfidfDim,
                    "nodeTextWeight": args.nodeTextWeight, "nodeTextDim": nodeTextDim, "nodeDim": nodeDim},
                   f, indent=2)
    print(f"\nSaved classifier to {args.checkpointDir}")


if __name__ == "__main__":
    main()
