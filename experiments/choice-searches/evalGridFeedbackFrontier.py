# Same harness again, this time with the spec's third display policy:
# FRONTIER -- instead of one fixed exploration weight applied to the whole
# grid (evalGridFeedbackUCB.py's kappa sweep), sweep the UCB exploration
# coefficient across geomspace(kappaLo, kappaHi, m) and take one argmax pick
# per value, so a single round's grid spans pure-exploit (small kappa) to
# pure-explore (large kappa) by construction. Dedupe collisions, backfill any
# remaining slots from a fixed-kappa UCB pass.
#
# This is a coarse, discrete-corpus version of what the batch-BO literature
# calls constructing one diverse batch per iteration instead of a single
# acquisition optimum -- see GP-UCB-PE (Contal et al., "Parallelizing
# Exploration-Exploitation Tradeoffs with Gaussian Process Bandit
# Optimization", ICML 2013: one exploit maximizer + multiple pure-exploration
# picks by posterior variance) and Local Penalization (Gonzalez et al.,
# "Batch Bayesian Optimization via Local Penalization", AISTATS 2016: greedily
# build a batch by penalizing the acquisition function near already-picked
# points). Frontier is a cruder version of the same idea -- instead of
# penalizing near picked points, it just varies the explore/exploit weight
# per slot -- but the underlying reason a single fixed kappa underperforms is
# the same one both papers are responding to: one exploitation maximizer
# alone re-samples the same region every round, and one exploration maximizer
# alone never commits, so a working batch needs a spread across the two.
#
# The kappaUCB.py finding this is testing: fixed kappa=1.0 at m=20,k=3,eps=0
# on the full 39421-font corpus produced a bimodal outcome -- when it escaped
# the absorbing basin it converged in median 4 rounds (mode 3, matching the
# small-corpus numbers almost exactly), but 58% of trials never escaped
# within 30 rounds at all. The question here is whether spreading the
# exploration weight within each round's grid (rather than fixing it) rescues
# more of that 58%.

import time

import numpy as np

from evalGridFeedback import loadCorpus, fitProjection, seedGrid, ACCEPTANCE_SET_SIZE, REGULARIZATION, MAX_ROUNDS
from evalGridFeedbackUCB import fitLaplace


def nextGridFrontier(Z, weight, covariance, shown, m, kappaLo, kappaHi, kappaBackfill):
    mu = Z @ weight
    ZSigma = Z @ covariance
    sigma = np.sqrt(np.clip(np.einsum('ij,ij->i', ZSigma, Z), 0, None))

    shownMask = np.zeros(Z.shape[0], dtype=bool)
    shownMask[list(shown)] = True

    kappas = np.geomspace(max(kappaLo, 1e-3), kappaHi, m)
    picks = []
    for kappa in kappas:
        ucb = mu + kappa * sigma
        ucb = np.where(shownMask, -np.inf, ucb)
        picks.append(int(np.argmax(ucb)))

    chosen = list(dict.fromkeys(picks))  # dedupe, keep first occurrence's order

    if len(chosen) < m:
        ucbBackfill = np.where(shownMask, -np.inf, mu + kappaBackfill * sigma)
        order = np.argsort(-ucbBackfill)
        chosenSet = set(chosen)
        for index in order:
            index = int(index)
            if index in chosenSet:
                continue
            chosen.append(index)
            chosenSet.add(index)
            if len(chosen) >= m:
                break

    return chosen[:m]


def runTrialFrontier(Z, cosineUnit, seed, m, k, t, epsilon, kappaLo, kappaHi, kappaBackfill,
                      targetIndex, rng, maxRounds, regularization):
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

        chosen = nextGridFrontier(Z, weight, covariance, shown, m, kappaLo, kappaHi, kappaBackfill)
        if not chosen:
            break

        shown.update(chosen)
        display = np.array(chosen)

    return None


def sweep(Z, cosineUnit, seed, m, k, epsilon, kappaHiValues, trials, kappaLo=1e-3,
          maxRounds=MAX_ROUNDS, t=ACCEPTANCE_SET_SIZE, regularization=REGULARIZATION, seedRng=0):
    results = {}
    for kappaHi in kappaHiValues:
        kappaBackfill = kappaHi  # backfill pass uses the "most explorative" tier, matches spec's policy B
        rng = np.random.default_rng(seedRng)
        rounds, censored = [], 0
        successRounds = []
        start = time.time()
        for trial in range(trials):
            targetIndex = int(rng.integers(0, Z.shape[0]))
            result = runTrialFrontier(Z, cosineUnit, seed, m, k, t, epsilon, kappaLo, kappaHi,
                                       kappaBackfill, targetIndex, rng, maxRounds, regularization)
            if result is None:
                censored += 1
                rounds.append(maxRounds)
            else:
                rounds.append(result)
                successRounds.append(result)
            print(f"\rkappaHi={kappaHi:>5.1f}  trial {trial + 1}/{trials}", end="")
        median = float(np.median(rounds))
        successMedian = float(np.median(successRounds)) if successRounds else float("nan")
        elapsed = time.time() - start
        results[kappaHi] = (median, censored / trials, successMedian, sorted(successRounds))
        print(f"\rkappaHi={kappaHi:>5.1f}  median={median:5.1f}  censored={censored}/{trials}  "
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

    kappaHiValues = [4.0, 8.0, 16.0, 32.0]
    print(f"\n=== Frontier kappaHi sweep at m={m} k={k} eps={epsilon} (kappaLo~0) ===")
    results = sweep(Z, cosineUnit, seed, m, k, epsilon, kappaHiValues, trials=50)

    print("\nkappaHi   median   censored   median(successes only)")
    for kappaHi in kappaHiValues:
        median, censoredFrac, successMedian, successList = results[kappaHi]
        print(f"{kappaHi:>7.1f}   {median:>6.1f}   {censoredFrac*100:>5.1f}%   {successMedian:>6.1f}"
              f"   n_success={len(successList)}  {successList}")


if __name__ == "__main__":
    main()
