"""
Checks a foundational question the user raised: this corpus's raw
embeddings are known (CLAUDE.md's pooling-bug analysis) to carry real
per-font norm variation (0.70-1.00) that's a DIFFERENT signal than
style direction (a font's own internal letter-to-letter consistency,
not stylistic content) -- yet the tree (sklearn KMeans, always
Euclidean), corpus.nearest's acceptance-set definition, rankInLeaf,
and oracleChoiceAmongChildren all use Euclidean distance in the
whitened space, not cosine. If that norm variation survives whitening,
Euclidean distance there conflates style similarity with an unrelated
per-font consistency signal -- a real methodological concern, not a
style nitpick, since it could affect the tree structure and every
recall@k number reported this session.

Checks, cheaply (no full NxN matrix):
  1. Does whitened-space norm variation actually persist (whitening
     doesn't force unit norm the way L2-normalization would -- but
     does it happen to end up nearly-uniform anyway)?
  2. For a sample of query fonts, how much do Euclidean-nearest-10 and
     cosine-nearest-10 (both in the whitened space) actually disagree
     in practice (Jaccard overlap)? Low overlap = the metric choice
     materially changes results; high overlap = the theoretical
     concern doesn't bite much for this specific corpus.

    python3 experiments/diffusion-searches/diagnose_metric_choice.py
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from corpus import FontCorpus
from dataset import PCAWhitener, EMBEDDINGS_PATH


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseCheckpoint", default="checkpoints/nav_classifier_v4")
    parser.add_argument("--sampleSize", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parseArgs()
    device = "cpu"

    import json
    with open(f"{args.baseCheckpoint}/config.json") as f:
        clsConfig = json.load(f)
    whitener = PCAWhitener.load(f"{clsConfig['baseCheckpoint']}/whitener.npz")
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device=device)

    norms = torch.norm(corpus.whitenedMatrix, dim=1).numpy()
    print(f"whitened-space vector norms: mean={norms.mean():.3f} std={norms.std():.3f} "
          f"min={norms.min():.3f} max={norms.max():.3f} "
          f"coefficient of variation={norms.std() / norms.mean():.4f}")
    print("(for reference: documented RAW-space norms were 0.70-1.00, i.e. a real ~30% relative spread)")

    rng = np.random.RandomState(args.seed)
    sampleIdx = rng.choice(len(corpus.names), min(args.sampleSize, len(corpus.names)), replace=False)
    queryVecs = corpus.whitenedMatrix[sampleIdx]

    # Euclidean (matches corpus.nearest / the tree's own construction metric)
    eucDists = torch.cdist(queryVecs, corpus.whitenedMatrix)
    eucTop = eucDists.topk(args.k + 1, largest=False).indices  # +1, index 0 will be self

    # Cosine (normalize both sides, then it's equivalent to nearest by cosine similarity)
    queryNorm = F.normalize(queryVecs, dim=1)
    corpusNorm = F.normalize(corpus.whitenedMatrix, dim=1)
    cosSims = queryNorm @ corpusNorm.T
    cosTop = cosSims.topk(args.k + 1, largest=True).indices

    overlaps = []
    for i, qi in enumerate(sampleIdx):
        eucSet = {t.item() for t in eucTop[i] if t.item() != qi}
        cosSet = {t.item() for t in cosTop[i] if t.item() != qi}
        eucSet = set(list(eucSet)[:args.k])
        cosSet = set(list(cosSet)[:args.k])
        overlap = len(eucSet & cosSet) / args.k
        overlaps.append(overlap)

    overlaps = np.array(overlaps)
    print(f"\nJaccard-style top-{args.k} overlap between Euclidean-nearest and cosine-nearest "
          f"(1.0 = identical rankings, {1.0/args.k:.3f} = chance-level for a random {args.k}-subset "
          f"of a large corpus): mean={overlaps.mean():.4f}  median={np.median(overlaps):.4f}  "
          f"min={overlaps.min():.4f}  (n={len(overlaps)} queries)")

    # does norm correlate with how much a font's neighbor set changes under the two metrics?
    sampleNorms = norms[sampleIdx]
    corr = np.corrcoef(sampleNorms, overlaps)[0, 1]
    print(f"correlation between a font's own norm and its Euclidean/cosine neighbor-set overlap: {corr:.4f}")


if __name__ == "__main__":
    main()
