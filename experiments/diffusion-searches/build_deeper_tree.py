"""
Builds a restructured hierarchical cluster tree, replacing the two
riskiest levels of the existing tree_variant_15_15_5_5_15.pkl with
several smaller-branching levels instead of just appending depth.

Why this specific change, per diagnose_top3_miss.py: depths 0 and 1
(both branching=15) cause 84% of all catastrophic top-3-miss failures
(the true child isn't even among the classifier's shown top-3, which
forecloses that session regardless of oracle noise or beam width).
Depth 4 ALSO has branching=15 but a LOW miss rate -- near-leaf groups
are tight/near-duplicate (this investigation's own corpus-density
findings), so even a hard 15-way choice is easy there. The risk isn't
branching factor in the abstract, it's branching factor paired with a
genuinely hard (broad, semantically-confusable) decision -- exactly
depths 0-1. And there's a sharp arithmetic reason smaller branching
specifically helps THOSE levels: showing top-3 of 15 covers only 20%
of possible answers; top-3 of a 4-way split covers 75% -- a top-3 miss
becomes structurally rarer independent of any accuracy improvement.

So: replace the two 15-branching levels with three 4-branching levels
(4^3=64, roughly comparable resolution to 15^2=225 while being much
safer per split), leaving the rest of the schedule (5, 5, 15) alone
since those levels aren't the problem. This ALSO sets up the
interleaved auto-descend logic in evaluate_classifier_branching.py
(fires whenever confidence clears --autoThreshold, at any depth) to
actually have somewhere to save real oracle-asks: with 6 levels now
against a 5-ask budget, a well-calibrated threshold auto-descending
through 1+ of the new safe/easy levels is required just to traverse
the whole tree within budget -- whatever's left over falls back to a
forced top-1 guess. Whether the restructuring alone (even under an
"always ask, guess the last unbudgeted level" policy) beats the old
tree should be measured BEFORE layering calibrated auto-descend on
top.

    python3 experiments/diffusion-searches/build_deeper_tree.py
"""
import time

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener

BASE_CHECKPOINT = "checkpoints/nav_classifier_v4"
OUT_PATH = "tree_variant_4_4_4_5_5_15.pkl"
BRANCHING = [4, 4, 4, 5, 5, 15]
MAX_DEPTH = 6
MIN_LEAF_SIZE = 5


def main():
    import json
    with open(f"{BASE_CHECKPOINT}/config.json") as f:
        clsConfig = json.load(f)
    from dataset import EMBEDDINGS_PATH
    whitener = PCAWhitener.load(f"{clsConfig['baseCheckpoint']}/whitener.npz")
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device="cpu")

    print(f"building tree: branching={BRANCHING}, maxDepth={MAX_DEPTH}, minLeafSize={MIN_LEAF_SIZE} ...")
    start = time.time()
    tree = HierarchicalClusterIndex.fit(corpus, branchingFactor=BRANCHING, maxDepth=MAX_DEPTH,
                                          minLeafSize=MIN_LEAF_SIZE)
    print(f"built in {time.time() - start:.1f}s")
    tree.save(OUT_PATH)
    print(f"saved to {OUT_PATH}")

    # quick sanity: leaf size distribution
    sizes = []
    def walk(node):
        if not node.children:
            sizes.append(len(node.memberIndices))
        else:
            for c in node.children:
                walk(c)
    walk(tree.root)
    import numpy as np
    sizes = np.array(sizes)
    print(f"{len(sizes)} leaves, size mean={sizes.mean():.1f} median={np.median(sizes):.0f} "
          f"min={sizes.min()} max={sizes.max()}")


if __name__ == "__main__":
    main()
