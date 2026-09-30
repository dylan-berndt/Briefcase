# GP-UCB-PE (Contal, Buffoni, Robicquet & Vayatis, "Parallelizing
# Exploration-Exploitation Tradeoffs with Gaussian Process Bandit
# Optimization", COLT/JMLR 2013), adapted from GP regression to the Laplace
# logistic posterior used by GridFeedbackSearch.
#
# The paper's batch construction, translated to this setting:
#   1. x_1 = argmax_unshown (mu(x) + kappa*sigma(x))          -- one UCB pick
#   2. "relevant region" R = {x : UCB(x) >= LCB(x_1)}         -- still could beat x_1
#   3. the remaining m-1 slots are PURE variance maximizers within R (no mu
#      term at all), with the posterior covariance fantasy-updated after each
#      pick -- using the *current* p(x) = sigmoid(w.x), not an actual label,
#      since (as in GP regression) a Laplace/Fisher covariance update from a
#      new query location doesn't require its label, only where it is -- so
#      picks 2..m don't collapse onto the same single highest-variance point.
#
# This is the one grounded alternative to the fixed-kappa UCB result (median
# 4 among successes, 58-63% censored) and to the spec's own bespoke Frontier
# policy (which turned out worse than fixed kappa -- see evalGridFeedbackFrontier.py).

import time

import numpy as np

from evalGridFeedback import loadCorpus, fitProjection, seedGrid, ACCEPTANCE_SET_SIZE, REGULARIZATION, MAX_ROUNDS
from evalGridFeedbackUCB import fitLaplace


def nextGridUCBPE(Z, weight, covariance, shown, m, kappa):
    N, D = Z.shape
    mu = Z @ weight
    ZSigma = Z @ covariance
    sigma = np.sqrt(np.clip(np.einsum('ij,ij->i', ZSigma, Z), 0, None))

    shownMask = np.zeros(N, dtype=bool)
    shownMask[list(shown)] = True

    ucb = mu + kappa * sigma
    lcb = mu - kappa * sigma
    maskedUCB = np.where(shownMask, -np.inf, ucb)
    firstIndex = int(np.argmax(maskedUCB))

    chosen = [firstIndex]
    chosenMask = shownMask.copy()
    chosenMask[firstIndex] = True

    relevant = (ucb >= lcb[firstIndex]) & (~chosenMask)

    p = 1.0 / (1.0 + np.exp(-(Z @ weight)))
    Hcur = np.linalg.inv(covariance)
    Sigma = covariance.copy()

    for _ in range(m - 1):
        candidates = np.where(relevant & ~chosenMask)[0]
        if len(candidates) == 0:
            candidates = np.where(~chosenMask & ~shownMask)[0]
            if len(candidates) == 0:
                break

        ZSigmaC = Z[candidates] @ Sigma
        sigma2C = np.clip(np.einsum('ij,ij->i', ZSigmaC, Z[candidates]), 0, None)
        pick = int(candidates[int(np.argmax(sigma2C))])

        chosen.append(pick)
        chosenMask[pick] = True

        z = Z[pick]
        a = p[pick] * (1.0 - p[pick])
        Hcur = Hcur + a * np.outer(z, z)
        Sigma = np.linalg.inv(Hcur)

    return chosen[:m]


def runTrialUCBPE(Z, cosineUnit, seed, m, k, t, epsilon, kappa, targetIndex, rng, maxRounds, regularization):
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

        chosen = nextGridUCBPE(Z, weight, covariance, shown, m, kappa)
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
        rounds, censored, successRounds = [], 0, []
        start = time.time()
        for trial in range(trials):
            targetIndex = int(rng.integers(0, Z.shape[0]))
            result = runTrialUCBPE(Z, cosineUnit, seed, m, k, t, epsilon, kappa,
                                    targetIndex, rng, maxRounds, regularization)
            if result is None:
                censored += 1
                rounds.append(maxRounds)
            else:
                rounds.append(result)
                successRounds.append(result)
            print(f"\rkappa={kappa:>5.2f}  trial {trial + 1}/{trials}", end="")
        median = float(np.median(rounds))
        successMedian = float(np.median(successRounds)) if successRounds else float("nan")
        elapsed = time.time() - start
        results[kappa] = (median, censored / trials, successMedian, sorted(successRounds))
        print(f"\rkappa={kappa:>5.2f}  median={median:5.1f}  censored={censored}/{trials}  "
              f"successMedian={successMedian:5.1f}  ({elapsed:.1f}s)" + " " * 10)
    return results


def main():
    print("Loading embeddings/all.json ...")
    names, V = loadCorpus()
    print(f"{len(names)} fonts")

    Z, D = fitProjection(V)
    print(f"Whitened subspace: D={D}")
    cosineUnit = V / np.linalg.norm(V, axis=1, keepdims=True)

    m, k, epsilon = 20, 3, 0.0
    print(f"\nComputing seed grid for m={m} ...")
    seed = seedGrid(Z, m)

    kappaValues = [0.5, 1.0, 2.0, 4.0]
    print(f"\n=== GP-UCB-PE kappa sweep at m={m} k={k} eps={epsilon} ===")
    results = sweep(Z, cosineUnit, seed, m, k, epsilon, kappaValues, trials=50)

    print("\nkappa   median   censored   median(successes)   n_success")
    for kappa in kappaValues:
        median, censoredFrac, successMedian, successList = results[kappa]
        print(f"{kappa:>5.2f}   {median:>6.1f}   {censoredFrac*100:>5.1f}%   {successMedian:>6.1f}"
              f"              {len(successList)}  {successList}")


if __name__ == "__main__":
    main()
