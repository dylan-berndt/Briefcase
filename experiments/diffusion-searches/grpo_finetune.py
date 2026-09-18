"""
GRPO fine-tuning of the text-conditioned diffusion MLP, warm-started from
an existing supervised checkpoint, directly optimizing OWN-TARGET
DISTANCE (the actual own-target-precision metric this whole investigation
has been chasing) instead of the proxy MSE-on-predicted-noise loss the
base model was trained with.

Formulation (DDPO -- Black et al. 2023 -- treats the reverse diffusion
chain as a T-step MDP; GRPO -- Shao et al. 2024 -- replaces a learned
value/critic baseline with a GROUP-RELATIVE advantage): at each reverse
step, state=(x_t, t, text), action=x_{t-1}, policy=N(mean_theta(x_t,t,
text), betaT). The Gaussian's log-density of the ACTUALLY SAMPLED action
is closed-form and differentiable w.r.t. theta through `mean`. A "group"
is G independent reverse-diffusion particles sharing one (font, query)
conditioning vector -- exactly the existing particle-batch mechanism
this project's branching search already uses, no new sampling
infrastructure needed. Reward = -||finalX - trueTarget|| (whitened
space); advantage = (reward - group_mean) / (group_std + eps), so no
separate value network is trained at all.

Two-phase per training step, standard for REINFORCE-style policy
gradients on a stochastic multi-step process: (1) roll out full
trajectories under the CURRENT (frozen at rollout time) policy, no_grad,
recording every step's (x_t, t, xPrev) and the final reward; (2)
recompute each step's `mean` with grad enabled (same weights as the
rollout -- this is an on-policy, single-gradient-step update per batch,
not multi-epoch PPO, so no importance-weighting/clipping is needed) and
take one gradient step on -advantage * sum_t log p(xPrev_t | x_t, t).
The deterministic final step (i==0, betaT effectively contributes no
noise) is excluded from the log-prob sum -- there's no randomness there
to attribute credit/blame to.

    python3 experiments/diffusion-searches/grpo_finetune.py \
        --initCheckpoint checkpoints/diffusion_lr_5e-4 \
        --checkpointDir checkpoints/diffusion_grpo
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PCAWhitener, QueryFontDataset, loadRaw, splitQueryCache
from diffusion import DiffusionMLP, TwoPhaseDiffusionMLP, GaussianDiffusion

LOG_2PI = math.log(2 * math.pi)


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initCheckpoint", required=True,
                         help="Existing supervised checkpoint dir to warm-start from (config.json, "
                              "checkpoint.pt, whitener.npz -- all reused as-is).")
    parser.add_argument("--checkpointDir", required=True)
    parser.add_argument("--groupSize", type=int, default=16, help="Particles per (font, query) group.")
    parser.add_argument("--batchExamples", type=int, default=8, help="Distinct (font, query) pairs per step.")
    parser.add_argument("--numSteps", type=int, default=20, help="Respaced reverse-diffusion steps per rollout.")
    parser.add_argument("--iterations", type=int, default=200, help="GRPO gradient steps.")
    parser.add_argument("--learningRate", type=float, default=1e-6,
                         help="Deliberately much lower than supervised training's 2e-4-5e-4 -- RL fine-"
                              "tuning a pretrained policy should nudge it, not overwrite it.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--logEvery", type=int, default=10)
    return parser.parse_args()


def loadModel(checkpointDir, config, device):
    if config.get("architecture", "standard") == "shapeI":
        model = TwoPhaseDiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"],
                                      hiddenDim=config["hiddenDim"], timeDim=config.get("timeDim", 64),
                                      numPlainBlocks=config.get("numPlainBlocks", 1),
                                      numConditionedBlocks=config.get("numConditionedBlocks", 2))
    else:
        model = DiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"],
                              hiddenDim=config["hiddenDim"], depth=config["depth"],
                              conditioning=config.get("conditioning", "concat"))
    model.load_state_dict(torch.load(os.path.join(checkpointDir, "checkpoint.pt"), map_location=device))
    return model.to(device)


def rollout(diffusion, model, text, visualDim, steps, device):
    """
    text: [N, textDim], N = batchExamples * groupSize.
    Returns finalX [N, visualDim] and a list of per-step dicts (x, t,
    xPrev, alphaBarT, prevAlphaBarT, betaT -- all needed to recompute the
    log-prob later with grad enabled), oldest-timestep-last i.e. in the
    order steps were actually taken (T -> 0).
    """
    N = text.shape[0]
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

            predictedNoise = model(x, t, text)
            mean = (1.0 / torch.sqrt(alphaT)) * (x - (betaT / torch.sqrt(1.0 - alphaBarT)) * predictedNoise)

            isLast = (i == 0)
            noise = torch.randn_like(x)
            xPrev = mean if isLast else mean + torch.sqrt(betaT) * noise

            trajectory.append({"x": x, "t": t, "xPrev": xPrev, "alphaBarT": alphaBarT,
                                "prevAlphaBarT": prevAlphaBar[i], "betaT": betaT, "isLast": isLast})
            x = xPrev
    return x, trajectory


def stepLogProb(model, step, text):
    """Recomputes `mean` WITH grad and returns log p(xPrev | x, t, text) per particle, or None on the
    deterministic final step (no randomness to credit/blame)."""
    if step["isLast"]:
        return None
    alphaT = 1.0 - step["betaT"]
    predictedNoise = model(step["x"], step["t"], text)
    mean = (1.0 / torch.sqrt(alphaT)) * (step["x"] - (step["betaT"] / torch.sqrt(1.0 - step["alphaBarT"])) * predictedNoise)
    var = step["betaT"]
    D = step["x"].shape[-1]
    sqError = ((step["xPrev"] - mean) ** 2).sum(dim=-1)
    return -0.5 * (sqError / var) - 0.5 * D * (LOG_2PI + torch.log(var))


def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    with open(os.path.join(args.initCheckpoint, "config.json")) as f:
        config = json.load(f)

    fontEmbeddings, sentenceCache = loadRaw(sentenceCachePath=cachePathFor(config["sentenceModel"]))
    names = sorted(fontEmbeddings.keys())
    trainCache, _ = splitQueryCache(sentenceCache, names, testFraction=config["testFraction"], seed=config["seed"])
    whitener = PCAWhitener.load(os.path.join(args.initCheckpoint, "whitener.npz"))

    trainSet = QueryFontDataset(trainCache, fontEmbeddings, whitener)
    trainLoader = DataLoader(trainSet, batch_size=args.batchExamples, shuffle=True,
                              collate_fn=QueryFontDataset.collate, drop_last=True)
    trainIter = iter(trainLoader)

    model = loadModel(args.initCheckpoint, config, device)
    diffusion = GaussianDiffusion(timesteps=config["timesteps"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learningRate)

    steps = diffusion.respacedSteps(args.numSteps)
    G = args.groupSize
    os.makedirs(args.checkpointDir, exist_ok=True)

    rewardHistory = []
    for iteration in range(args.iterations):
        try:
            batch = next(trainIter)
        except StopIteration:
            trainIter = iter(trainLoader)
            batch = next(trainIter)

        text = batch["text"].to(device)          # [B, textDim]
        target = batch["visual"].to(device)      # [B, visualDim]
        B = text.shape[0]

        textGroup = text.repeat_interleave(G, dim=0)      # [B*G, textDim]
        targetGroup = target.repeat_interleave(G, dim=0)  # [B*G, visualDim]

        model.eval()
        finalX, trajectory = rollout(diffusion, model, textGroup, config["visualDim"], steps, device)

        dist = torch.norm(finalX - targetGroup, dim=-1)   # [B*G]
        reward = -dist
        rewardByGroup = reward.view(B, G)
        advantage = (rewardByGroup - rewardByGroup.mean(dim=1, keepdim=True)) / (rewardByGroup.std(dim=1, keepdim=True) + 1e-6)
        advantage = advantage.view(B * G).detach()

        model.train()
        optimizer.zero_grad()
        totalLogProb = torch.zeros(B * G, device=device)
        for step in trajectory:
            logProb = stepLogProb(model, step, textGroup)
            if logProb is not None:
                totalLogProb = totalLogProb + logProb

        loss = -(advantage * totalLogProb).mean()
        loss.backward()
        optimizer.step()

        rewardHistory.append(-dist.mean().item())
        if (iteration + 1) % args.logEvery == 0:
            recent = np.mean(rewardHistory[-args.logEvery:])
            print(f"iter {iteration + 1}/{args.iterations}  meanOwnTargetDist(recent)={-recent:.4f}  "
                  f"loss={loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(args.checkpointDir, "checkpoint.pt"))
    whitener.save(os.path.join(args.checkpointDir, "whitener.npz"))
    with open(os.path.join(args.initCheckpoint, "test_pairs.json")) as f:
        testPairs = json.load(f)
    with open(os.path.join(args.checkpointDir, "test_pairs.json"), "w") as f:
        json.dump(testPairs, f)

    newConfig = dict(config)
    newConfig.update({"grpoBase": args.initCheckpoint, "grpoGroupSize": G, "grpoNumSteps": args.numSteps,
                       "grpoIterations": args.iterations, "grpoLearningRate": args.learningRate})
    with open(os.path.join(args.checkpointDir, "config.json"), "w") as f:
        json.dump(newConfig, f, indent=2)

    print(f"Saved GRPO-finetuned checkpoint to {args.checkpointDir}")


if __name__ == "__main__":
    main()
