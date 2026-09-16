# Same Monte Carlo harness as evalGridFeedback.py, but the aggregator uses
# the original (pre-simplification) spec's Laplace-approximated logistic
# posterior + UCB display policy instead of pure top-m argmax:
#
#   H = sum_j p_j(1-p_j) z_j z_j^T + lambda I     (p_j = sigmoid(w.z_j))
#   Sigma = H^-1
#   UCB(v) = w.v + kappa * sqrt(v^T Sigma v)
#   D_{r+1} = top-m unshown by UCB(v)
#
# Motivation: evalGridFeedback.py's full-corpus sweep found pure top-m
# majority-censors even at zero noise -- the fit commits to one region and
# (per traced runs) the target's rank under the fitted weight just wanders
# rather than trending down. That's the textbook "absorbing basin" pure
# exploitation failure; UCB's uncertainty term is supposed to force sampling
# outside the current best region until the posterior actually narrows there.
# This script checks whether it does, and by how much.

import time

import numpy as np

from utils.search import GridFeedbackSearch
from evalGridFeedback import loadCorpus, fitProjection, seedGrid, ACCEPTANCE_SET_SIZE, REGULARIZATION, MAX_ROUNDS


def fitLaplace(Z, y, regularization):
    """Laplace approximation around the MAP logistic fit: weight + posterior covariance."""
    weight = GridFeedbackSearch._fitLogistic(Z, y, regularization)
    p = 1.0 / (1.0 + np.exp(-(Z @ weight)))
    w = p * (1.0 - p)
    H = (Z * w[:, None]).T @ Z + regularization * np.eye(Z.shape[1])
    covariance = np.linalg.inv(H)
    return weight, covariance


def runTrialUCB(Z, cosineUnit, seed, m, k, t, epsilon, kappa, targetIndex, rng, maxRounds, regularization):
    N, D = Z.shape
    simFull = cosineUnit @ cosineUnit[targetIndex]
    acceptanceSet = set(int(i) for i in np.argsort(-simFull)[:t])

    shown = set(int(i) for i in seed)
    display = seed
    labeledIndices, labels = [], []

    for round in range(1, maxRounds + 1):
        if acceptanceSet.intersection(int(i) for i in display):
            return round

        order = np.argsort(-simFull[display])
        trueTop = set(int(display[i]) for i in order[:k])
        for index in display:
            index = int(index)
            label = 1 if index in trueTop else 0
            if rng.random() < epsilon:
                label = 1 - label
            labeledIndices.append(index)
            labels.append(label)

        y = np.array(labels)
        if len(np.unique(y)) < 2:
            positives = [Z[i] for i, l in zip(labeledIndices, labels) if l == 1]
            weight = np.mean(positives, axis=0) if positives else np.zeros(D)
            covariance = np.eye(D) / regularization
        else:
            signed = np.where(y == 1, 1.0, -1.0)
            weight, covariance = fitLaplace(Z[labeledIndices], signed, regularization)

        mu = Z @ weight
        ZSigma = Z @ covariance
        sigma2 = np.clip(np.einsum('ij,ij->i', ZSigma, Z), 0, None)
        ucb = mu + kappa * np.sqrt(sigma2)

        order = np.argsort(-ucb)
        chosen = []
        for index in order:
            index = int(index)
            if index in shown:
                continue
            chosen.append(index)
            if len(chosen) >= m:
                break

        if not chosen:
            break

        shown.update(chosen)
        display = np.array(chosen)

    return None


def sweep(Z, cosineUnit, seed, m, k, epsilon, kappaValues, trials, maxRounds=MAX_ROUNDS,
          t=ACCEPTANCE_SET_SIZE, regularization=REGULARIZATION, seedRng=0):
    results = {}
    for kappa in kappaValues:
        rng = np.random.default_rng(seedRng)
        rounds, censored = [], 0
        start = time.time()
        for trial in range(trials):
            targetIndex = int(rng.integers(0, Z.shape[0]))
            result = runTrialUCB(Z, cosineUnit, seed, m, k, t, epsilon, kappa,
                                  targetIndex, rng, maxRounds, regularization)
            if result is None:
                censored += 1
                rounds.append(maxRounds)
            else:
                rounds.append(result)
            print(f"\rkappa={kappa:>5.2f}  trial {trial + 1}/{trials}", end="")
        median = float(np.median(rounds))
        elapsed = time.time() - start
        results[kappa] = (median, censored / trials)
        print(f"\rkappa={kappa:>5.2f}  median={median:5.1f}  censored={censored}/{trials}  "
              f"({elapsed:.1f}s)" + " " * 10)
    return results


def main():
    import sys

    print("Loading embeddings/all.json ...")
    names, V = loadCorpus()

    if len(sys.argv) > 1:
        subsetSize = int(sys.argv[1])
        rng = np.random.default_rng(1)
        subsetIndex = rng.choice(len(names), size=subsetSize, replace=False)
        names, V = names[subsetIndex], V[subsetIndex]
        print(f"Subsampled to {len(names)} fonts")
    else:
        print(f"{len(names)} fonts")

    Z, D = fitProjection(V)
    print(f"Whitened subspace: D={D}")
    cosineUnit = V / np.linalg.norm(V, axis=1, keepdims=True)

    m, k, epsilon = 20, 3, 0.0
    print(f"\nComputing seed grid for m={m} ...")
    seed = seedGrid(Z, m)

    kappaValues = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    print(f"\n=== kappa sweep at m={m} k={k} eps={epsilon} ===")
    results = sweep(Z, cosineUnit, seed, m, k, epsilon, kappaValues, trials=30)

    print("\nkappa   median   censored")
    for kappa in kappaValues:
        median, censoredFrac = results[kappa]
        print(f"{kappa:>5.2f}   {median:>6.1f}   {censoredFrac*100:>5.1f}%")


if __name__ == "__main__":
    main()
