# Monte Carlo evaluation of GridFeedbackSearch: median rounds to surface a
# simulated user's target font, swept over grid size (m), choices made per
# round (k), and label noise (epsilon).
#
# Simulated user: target f* uniform over the corpus, acceptance set
# T = kNN_t(f*) (the user is happy with anything close, not just the exact
# font -- matches the design doc's evaluation, not a literal single-point
# match, which at N~39k would be a much harder and less meaningful task).
# Each round, the user's "true" picks are the k displayed fonts closest to
# f* by cosine similarity (in the embeddings/all.json space -- not the
# aggregator's internal whitened PCA space, so this also exercises the
# metric-mismatch the real aggregator would face). Each of the m displayed
# labels is flipped independently with probability epsilon. Success =
# display intersect T != empty (checked before labeling, same as a real
# "accept" ending the round).
#
# Uses embeddings/all.json directly instead of GridFeedbackSearch's own
# raw-feature cache, so no model or font dataset needs to be loaded -- just
# the offline PCA/whiten/seed-grid pipeline (mirrored here, since that part
# is normally bound up in GridFeedbackSearch.__init__) plus the class's
# already-static _fitLogistic and _participationRatio.

import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from utils.search import GridFeedbackSearch

REGULARIZATION = 1.0
MAX_ROUNDS = 30
TRIALS_PER_COMBO = 80
SEED = 0

# Acceptance-set size: the user succeeds if any of the target's T nearest
# cosine neighbors shows up, not just the exact font. Matches the design
# doc's t=10 (its dominant, unswept parameter -- see its section 7).
ACCEPTANCE_SET_SIZE = 10

M_VALUES = [10, 20, 40]
K_VALUES = [1, 3, 5]
EPSILON_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]


def loadCorpus(path="embeddings/all.json"):
    with open(path, "r") as file:
        data = json.load(file)
    names = np.array(list(data.keys()))
    V = np.stack([np.array(data[name], dtype=np.float64) for name in names], axis=0)
    return names, V


def fitProjection(V, D=None):
    """Centre -> PCA(D) -> whiten, same steps as GridFeedbackSearch._fitProjection."""
    if D is None:
        D = GridFeedbackSearch._participationRatio(V)
        D = max(1, min(D, V.shape[0] - 1, V.shape[1]))

    centered = V - V.mean(axis=0)
    pca = PCA(n_components=D)
    projected = pca.fit_transform(centered)

    std = projected.std(axis=0)
    std[std < 1e-8] = 1e-8
    return projected / std, D


def seedGrid(Z, m, randomState=0):
    """k-means(Z, m) -> medoid of each cluster, same as GridFeedbackSearch._seedGrid."""
    clusters = min(m, Z.shape[0])
    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=randomState).fit(Z)

    seed = []
    for c in range(clusters):
        members = np.where(kmeans.labels_ == c)[0]
        if len(members) == 0:
            continue
        distances = np.linalg.norm(Z[members] - kmeans.cluster_centers_[c], axis=1)
        seed.append(members[np.argmin(distances)])
    return np.array(seed)


def runTrial(Z, cosineUnit, seed, m, k, t, epsilon, targetIndex, rng, maxRounds, regularization):
    N, D = Z.shape

    # Cosine similarity of every corpus font to the target, computed once per
    # trial (target is fixed for the whole session) and reused for both the
    # acceptance-set membership check and each round's "true" top-k labels.
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
        else:
            signed = np.where(y == 1, 1.0, -1.0)
            weight = GridFeedbackSearch._fitLogistic(Z[labeledIndices], signed, regularization)

        scores = Z @ weight
        order = np.argsort(-scores)
        chosen = []
        for index in order:
            index = int(index)
            if index in shown:
                continue
            chosen.append(index)
            if len(chosen) >= m:
                break

        if not chosen:
            break  # exhausted the corpus

        shown.update(chosen)
        display = np.array(chosen)

    return None  # censored


def main():
    import sys

    print("Loading embeddings/all.json ...")
    names, V = loadCorpus()

    if len(sys.argv) > 1:
        subsetSize = int(sys.argv[1])
        rng = np.random.default_rng(1)
        subsetIndex = rng.choice(len(names), size=subsetSize, replace=False)
        names, V = names[subsetIndex], V[subsetIndex]
        print(f"Subsampled to {len(names)} fonts, dim {V.shape[1]}")
    else:
        print(f"{len(names)} fonts, dim {V.shape[1]}")

    Z, D = fitProjection(V)
    print(f"Whitened subspace: D={D}")

    cosineUnit = V / np.linalg.norm(V, axis=1, keepdims=True)

    results = {}
    for m in M_VALUES:
        print(f"\nComputing seed grid for m={m} ...")
        seed = seedGrid(Z, m)

        for k in K_VALUES:
            if k > m:
                continue
            for epsilon in EPSILON_VALUES:
                rng = np.random.default_rng(SEED)
                rounds = []
                censored = 0
                start = time.time()

                for trial in range(TRIALS_PER_COMBO):
                    targetIndex = int(rng.integers(0, len(names)))
                    result = runTrial(Z, cosineUnit, seed, m, k, ACCEPTANCE_SET_SIZE, epsilon,
                                       targetIndex, rng, MAX_ROUNDS, REGULARIZATION)
                    if result is None:
                        censored += 1
                        rounds.append(MAX_ROUNDS)
                    else:
                        rounds.append(result)
                    print(f"\rm={m:>3} k={k:>2} eps={epsilon:.1f}  "
                          f"trial {trial + 1}/{TRIALS_PER_COMBO}", end="")

                median = float(np.median(rounds))
                elapsed = time.time() - start
                results[(m, k, epsilon)] = (median, censored / TRIALS_PER_COMBO)
                print(f"\rm={m:>3} k={k:>2} eps={epsilon:.1f}  "
                      f"median={median:5.1f}  censored={censored}/{TRIALS_PER_COMBO}  "
                      f"({elapsed:.1f}s)" + " " * 10)

    print("\n\n=== Summary: median rounds to find target (censored %) ===")
    header = "m".rjust(4) + "k".rjust(4) + "".join(f"eps={e:.1f}".rjust(16) for e in EPSILON_VALUES)
    print(header)
    for m in M_VALUES:
        for k in K_VALUES:
            if k > m:
                continue
            row = f"{m:>4}{k:>4}"
            for epsilon in EPSILON_VALUES:
                median, censoredFrac = results[(m, k, epsilon)]
                row += f"{median:>10.1f} ({censoredFrac*100:>3.0f}%)"
            print(row)


if __name__ == "__main__":
    main()
