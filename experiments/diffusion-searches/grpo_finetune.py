"""
GRPO fine-tuning of the text-conditioned diffusion MLP toward the actual
interactive-search task: given a text query AND the accumulated decision
history of a branching session, learn to route particles into the
CORRECT child of the corpus's hierarchical partition -- not just to
minimize single-shot embedding distance.

This is deliberately NOT single-shot RL fine-tuning (an earlier version
of this script did that and measured no improvement -- see git history;
that tested the wrong thing). The actual point, per the design this
implements: a diffusion model conditioned only on text can't do better
than its own per-level branch accuracy (measured ~32-34% pooled top-1
this investigation), and the interactive search's hard, irreversible
per-round commitment (branching.py's pruneAndResample) then compounds
that into a much worse end-to-end result (see the compounding-error
diagnostic: predicted and measured final survival matched almost
exactly from the product of per-depth survival rates). Conditioning the
model on WHERE THE SESSION HAS ALREADY NARROWED TO -- not just the
original text -- lets it learn from real (including wrong) navigation
history instead of only ever seeing the root-level problem.

Conditioning: the base model's text vector is augmented with the CURRENT
tree node's centroid (already in the same whitened space the model
operates in -- no new representation to learn). The warm-started model's
textProjection input layer is surgically widened (extra columns
zero-initialized) so it starts EXACTLY equivalent to the base checkpoint
at iteration 0, then GRPO training teaches it to actually use the new
signal.

Rollout ("trained on the noisy oracle"): for each (font, query) training
example, walk the REAL hierarchical corpus tree from the root using the
KNOWN target's true path. At each level: condition on (text, current
node centroid), roll out G particles (short respaced reverse diffusion,
DDPO-style MDP -- see the closed-form Gaussian log-prob math below),
assign each particle to the nearest CHILD of the current node. Reward is
CORRECTNESS, not distance, per your direction: +1 if a particle's
assigned child is the target's true child, -1 otherwise -- this directly
optimizes "does the model send particles to the right region," which is
what the search actually needs. GRPO's group-relative advantage
(normalize reward within the G particles sharing this decision point,
no value network) turns that into a policy gradient. The walk then
advances to the TRUE child with probability (1-noiseProb) or a random
WRONG child with probability noiseProb, simulating a real, imperfect
oracle -- so the model also sees, and has to cope with, being on a
wrong branch sometimes, not only ever the correct path.

    python3 experiments/diffusion-searches/grpo_finetune.py \
        --initCheckpoint checkpoints/diffusion_lr_5e-4 \
        --checkpointDir checkpoints/diffusion_grpo \
        --learningRate 1e-5
"""
import argparse
import json
import math
import os
import pickle

import numpy as np
import torch

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, loadRaw, splitQueryCache, EMBEDDINGS_PATH
from diffusion import DiffusionMLP, GaussianDiffusion

LOG_2PI = math.log(2 * math.pi)


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initCheckpoint", required=True)
    parser.add_argument("--checkpointDir", required=True)
    parser.add_argument("--treeCache", default="tree_variant_15_15_5_5_15.pkl",
                         help="Hierarchical corpus partition to navigate -- defaults to the validated "
                              "variable-branching-factor tree ([15,15,5,5,15]), not the uniform-10 one.")
    parser.add_argument("--groupSize", type=int, default=16, help="Particles per decision point.")
    parser.add_argument("--batchExamples", type=int, default=4, help="Distinct (font, query) pairs per step.")
    parser.add_argument("--numSteps", type=int, default=15, help="Respaced reverse-diffusion steps per rollout.")
    parser.add_argument("--maxDepth", type=int, default=5, help="Tree levels walked per training example.")
    parser.add_argument("--noiseProb", type=float, default=0.2,
                         help="Probability of advancing to a random WRONG child instead of the true one "
                              "after each level, simulating a real, imperfect oracle.")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learningRate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--logEvery", type=int, default=10)
    return parser.parse_args()


def widenTextProjection(stateDict, oldTextDim, extraDim):
    """
    Surgically widens textProjection's input layer from oldTextDim to
    oldTextDim+extraDim, zero-initializing the new columns so the warm-
    started model is EXACTLY equivalent to the base checkpoint at
    iteration 0 -- the new node-centroid conditioning contributes nothing
    until GRPO training actually shapes those weights.
    """
    stateDict = dict(stateDict)
    oldWeight = stateDict["textProjection.0.weight"]  # [hiddenDim, oldTextDim]
    hiddenDim = oldWeight.shape[0]
    newWeight = torch.zeros(hiddenDim, oldTextDim + extraDim, dtype=oldWeight.dtype)
    newWeight[:, :oldTextDim] = oldWeight
    stateDict["textProjection.0.weight"] = newWeight
    return stateDict


def rollout(diffusion, model, cond, visualDim, steps, device):
    """Same DDPO-style rollout as before: returns finalX and the per-step trajectory needed to
    recompute log-probs with grad later."""
    N = cond.shape[0]
    stepsTensor = torch.tensor(steps, device=diffusion.alphaBars.device)
    alphaBarAtStep = diffusion.alphaBars[stepsTensor]
    prevAlphaBar = torch.cat([torch.ones(1, device=alphaBarAtStep.device), alphaBarAtStep[:-1]])

    x = torch.randn(N, visualDim, device=device)
    trajectory = []
    with torch.no_grad():
        for i in reversed(range(len(steps))):
            t = torch.full((N,), steps[i], device=device, dtype=torch.long)
            alphaBarT = alphaBarAtStep[i]
            betaT = 1.0 - (alphaBarT / prevAlphaBar[i])
            alphaT = 1.0 - betaT

            predictedNoise = model(x, t, cond)
            mean = (1.0 / torch.sqrt(alphaT)) * (x - (betaT / torch.sqrt(1.0 - alphaBarT)) * predictedNoise)

            isLast = (i == 0)
            noise = torch.randn_like(x)
            xPrev = mean if isLast else mean + torch.sqrt(betaT) * noise

            trajectory.append({"x": x, "t": t, "xPrev": xPrev, "alphaBarT": alphaBarT,
                                "prevAlphaBarT": prevAlphaBar[i], "betaT": betaT, "isLast": isLast})
            x = xPrev
    return x, trajectory


def stepLogProb(model, step, cond):
    if step["isLast"]:
        return None
    alphaT = 1.0 - step["betaT"]
    predictedNoise = model(step["x"], step["t"], cond)
    mean = (1.0 / torch.sqrt(alphaT)) * (step["x"] - (step["betaT"] / torch.sqrt(1.0 - step["alphaBarT"])) * predictedNoise)
    var = step["betaT"]
    D = step["x"].shape[-1]
    sqError = ((step["xPrev"] - mean) ** 2).sum(dim=-1)
    return -0.5 * (sqError / var) - 0.5 * D * (LOG_2PI + torch.log(var))


def trueChildAt(node, targetIdx):
    for i, child in enumerate(node.children):
        if targetIdx in child.memberIndices:
            return i, child
    return None, None


def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    with open(os.path.join(args.initCheckpoint, "config.json")) as f:
        config = json.load(f)

    fontEmbeddings, sentenceCache = loadRaw(sentenceCachePath=cachePathFor(config["sentenceModel"]))
    names = sorted(fontEmbeddings.keys())
    trainCache, _ = splitQueryCache(sentenceCache, names, testFraction=config["testFraction"], seed=config["seed"])
    whitener = PCAWhitener.load(os.path.join(args.initCheckpoint, "whitener.npz"))
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device=device)
    hier = HierarchicalClusterIndex.load(args.treeCache)

    trainPairs = [(name, i) for name, vectors in trainCache.items() for i in range(len(vectors))
                  if name in corpus.nameToIndex]
    print(f"{len(trainPairs)} training (font, query) pairs available")

    baseTextDim = config["textDim"]
    pcaDim = config["visualDim"]
    newTextDim = baseTextDim + pcaDim

    baseStateDict = torch.load(os.path.join(args.initCheckpoint, "checkpoint.pt"), map_location=device)
    widenedStateDict = widenTextProjection(baseStateDict, baseTextDim, pcaDim)

    model = DiffusionMLP(visualDim=config["visualDim"], textDim=newTextDim, hiddenDim=config["hiddenDim"],
                          depth=config["depth"], conditioning=config.get("conditioning", "concat")).to(device)
    model.load_state_dict(widenedStateDict)

    diffusion = GaussianDiffusion(timesteps=config["timesteps"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learningRate)
    steps = diffusion.respacedSteps(args.numSteps)
    G = args.groupSize
    os.makedirs(args.checkpointDir, exist_ok=True)

    correctnessHistory = []
    for iteration in range(args.iterations):
        batchIdx = rng.choice(len(trainPairs), args.batchExamples, replace=False)
        optimizer.zero_grad()
        totalLoss = 0.0
        numLevels = 0
        correctCount, totalCount = 0, 0

        for idx in batchIdx:
            name, qi = trainPairs[idx]
            targetIdx = corpus.nameToIndex[name]
            textVec = torch.from_numpy(np.asarray(sentenceCache[name][qi], dtype=np.float32)).to(device)

            node = hier.root
            depth = 0
            while node.children and depth < args.maxDepth:
                trueIdx, trueChild = trueChildAt(node, targetIdx)
                if trueIdx is None:
                    break

                centroid = torch.from_numpy(node.centroid.astype(np.float32)).to(device)
                cond = torch.cat([textVec, centroid]).unsqueeze(0).expand(G, -1).contiguous()

                model.eval()
                finalX, trajectory = rollout(diffusion, model, cond, config["visualDim"], steps, device)

                childCentroids = torch.from_numpy(
                    np.stack([c.centroid for c in node.children]).astype(np.float32)).to(device)
                assignments = torch.cdist(finalX, childCentroids).argmin(dim=1)  # [G]
                correct = (assignments == trueIdx)
                reward = torch.where(correct, torch.ones(G, device=device), -torch.ones(G, device=device))
                advantage = (reward - reward.mean()) / (reward.std() + 1e-6)
                advantage = advantage.detach()

                correctCount += correct.sum().item()
                totalCount += G

                model.train()
                totalLogProb = torch.zeros(G, device=device)
                for step in trajectory:
                    logProb = stepLogProb(model, step, cond)
                    if logProb is not None:
                        totalLogProb = totalLogProb + logProb
                levelLoss = -(advantage * totalLogProb).mean()
                totalLoss = totalLoss + levelLoss
                numLevels += 1

                if rng.random() < args.noiseProb and len(node.children) > 1:
                    wrongOptions = [i for i in range(len(node.children)) if i != trueIdx]
                    node = node.children[rng.choice(wrongOptions)]
                else:
                    node = trueChild
                depth += 1

        if numLevels > 0:
            (totalLoss / numLevels).backward()
            optimizer.step()

        acc = correctCount / max(totalCount, 1)
        correctnessHistory.append(acc)
        if (iteration + 1) % args.logEvery == 0:
            recent = np.mean(correctnessHistory[-args.logEvery:])
            print(f"iter {iteration + 1}/{args.iterations}  particle-correct-child rate (recent)={recent:.3f}  "
                  f"loss={(totalLoss / max(numLevels,1)).item():.4f}")

    torch.save(model.state_dict(), os.path.join(args.checkpointDir, "checkpoint.pt"))
    whitener.save(os.path.join(args.checkpointDir, "whitener.npz"))
    with open(os.path.join(args.initCheckpoint, "test_pairs.json")) as f:
        testPairs = json.load(f)
    with open(os.path.join(args.checkpointDir, "test_pairs.json"), "w") as f:
        json.dump(testPairs, f)

    newConfig = dict(config)
    newConfig.update({"textDim": newTextDim, "baseTextDim": baseTextDim, "grpoBase": args.initCheckpoint,
                       "grpoDecisionAware": True, "grpoTreeCache": args.treeCache,
                       "grpoGroupSize": G, "grpoNumSteps": args.numSteps, "grpoMaxDepth": args.maxDepth,
                       "grpoNoiseProb": args.noiseProb, "grpoIterations": args.iterations,
                       "grpoLearningRate": args.learningRate})
    with open(os.path.join(args.checkpointDir, "config.json"), "w") as f:
        json.dump(newConfig, f, indent=2)

    print(f"Saved decision-aware GRPO-finetuned checkpoint to {args.checkpointDir}")


if __name__ == "__main__":
    main()
