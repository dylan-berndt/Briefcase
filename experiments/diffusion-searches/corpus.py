"""
Loads the font-embedding corpus and exposes it in the SAME PCA-whitened
space diffusion.py's reverse process natively operates in, so branching.py
/ oracle.py / evaluate_branching.py never need to round-trip a generated
sample back out to the raw 512-d ambient space (inverseTransform, then
re-normalize) just to compare it against real fonts -- that round trip
was measured to be roughly metric-equivalent to comparing directly in the
whitened space anyway, and doing it directly is both cheaper and avoids a
real scale mismatch: fitting/applying PCAWhitener against an L2-NORMALIZED
copy of the corpus (as an earlier version of this module did for
ClusterIndex) doesn't match the space PCAWhitener.fit was actually built
from (dataset.py fits it on the RAW, un-normalized embeddings, whose norms
in [0.72, 1.0] carry real per-font information -- see utils/embeddings.py's
documented pooling-normalization bug).

Also provides HierarchicalClusterIndex: a fixed, one-time hierarchical
k-means partition of the corpus in that whitened space, which
branching.py's decision-point detection navigates round by round. See its
docstring for why this replaced a single flat (non-hierarchical) partition.
"""
import pickle

import numpy as np
import torch

from dataset import loadFontEmbeddings, EMBEDDINGS_PATH


class FontCorpus:
    def __init__(self, names, whitenedMatrix, device):
        self.names = names
        self.whitenedMatrix = whitenedMatrix  # [N, pcaDim], torch, on `device` -- whitener.transform(rawEmbeddings)
        self.nameToIndex = {name: i for i, name in enumerate(names)}
        self.device = device

    @staticmethod
    def load(whitener, path=EMBEDDINGS_PATH, device="cpu"):
        fontEmbeddings = loadFontEmbeddings(path)
        names = sorted(fontEmbeddings.keys())
        raw = np.stack([fontEmbeddings[n] for n in names], axis=0)
        whitened = whitener.transform(raw)
        whitenedT = torch.from_numpy(whitened.astype(np.float32)).to(device)
        return FontCorpus(names, whitenedT, device)

    def nearest(self, vectors, k=1, restrictTo=None):
        """
        vectors: [B, pcaDim], in this corpus's whitened space (e.g. a
        diffusion sample or x0Hat estimate).
        restrictTo: optional 1-D array/tensor of corpus indices to search
        within instead of the whole corpus (e.g. one hierarchy node's own
        real member fonts).
        Returns (distances [B, k] ascending, names -- a [B][k] list of lists).
        """
        candidates = self.whitenedMatrix if restrictTo is None else self.whitenedMatrix[restrictTo]
        dists = torch.cdist(vectors, candidates)
        k = min(k, dists.shape[1])
        top = dists.topk(k, dim=1, largest=False)
        if restrictTo is None:
            names = [[self.names[i] for i in row] for row in top.indices.tolist()]
        else:
            restrictTo = restrictTo if torch.is_tensor(restrictTo) else torch.as_tensor(restrictTo)
            names = [[self.names[restrictTo[i].item()] for i in row] for row in top.indices.tolist()]
        return top.values, names

    def rank(self, vectors):
        """Full ascending-distance ranking of every candidate for each row of vectors. Returns (indices [B, N], dists [B, N])."""
        dists = torch.cdist(vectors, self.whitenedMatrix)
        return dists.argsort(dim=1), dists


class ClusterNode:
    def __init__(self, centroid, memberIndices, children):
        self.centroid = centroid            # [pcaDim] numpy
        self.memberIndices = memberIndices  # np.array of corpus indices under this node (all descendants)
        self.children = children            # list[ClusterNode], empty at leaves


class HierarchicalClusterIndex:
    """
    A fixed, one-time hierarchical k-means partition of the font corpus
    in its whitened space, replacing an earlier flat (single-level,
    reused-every-round) partition.

    Why hierarchical: a flat k-way partition caps addressable resolution
    at k no matter how many rounds are spent -- once particles settle
    into one of the k regions, re-checking against the SAME global
    centroids doesn't narrow any further within it. A depth-D,
    branching-B tree can in principle distinguish up to B^D leaves in D
    rounds (e.g. depth 5 x branching 10 = 100,000, well above this
    corpus's 39,421 fonts). This was measured, not assumed: navigating
    this exact tree with an OMNISCIENT oracle (no diffusion model
    involved at all, just always stepping into the child that truly
    contains a known target) got a median final leaf of 4 fonts in <=5
    rounds -- implied recall@10 ~97% -- versus the ~10% recall@10 the
    flat-partition version of this search measured. The flat partition,
    not the embedding space, was the bottleneck.

    Building the tree is a real one-time cost (~90s for the defaults
    below over the full corpus on this machine) -- see save/load.
    """

    def __init__(self, root):
        self.root = root

    @staticmethod
    def fit(corpus, branchingFactor=10, maxDepth=5, minLeafSize=5, randomState=0):
        """
        branchingFactor: either a single int (uniform fanout at every
        depth, original behavior) or a list/tuple giving a PER-DEPTH
        fanout (index 0 = root's children, etc.; the last entry repeats
        for any depth beyond the list's length). Motivation for the
        per-depth form: measured per-level model accuracy (tree_dip_
        investigation.py) is decent at the root and near leaves but
        worst at middle depths (2-3) -- plausibly because a 10-way
        choice among already-fine-grained, similar-but-distinct style
        groups is a harder categorical decision than a 10-way choice
        among broad macro-categories. A SMALLER fanout at exactly those
        depths turns each individual decision there into an easier
        (fewer-way) categorical choice, at the cost of needing more
        depth to reach the same total addressable resolution -- an
        explicit, testable trade against the uniform baseline, not
        assumed to help.
        """
        from sklearn.cluster import KMeans, MiniBatchKMeans

        whitened = corpus.whitenedMatrix.cpu().numpy()

        def branchingAt(depth):
            if isinstance(branchingFactor, (list, tuple)):
                return branchingFactor[min(depth, len(branchingFactor) - 1)]
            return branchingFactor

        def buildNode(indices, depth):
            centroid = whitened[indices].mean(axis=0)
            if depth >= maxDepth or len(indices) <= minLeafSize:
                return ClusterNode(centroid, indices, [])
            k = min(branchingAt(depth), max(2, len(indices) // 3))
            estimator = (MiniBatchKMeans(n_clusters=k, n_init=3, random_state=randomState, batch_size=2048)
                         if len(indices) > 5000 else KMeans(n_clusters=k, n_init=3, random_state=randomState))
            labels = estimator.fit_predict(whitened[indices])
            children = [buildNode(indices[labels == c], depth + 1)
                        for c in range(k) if np.any(labels == c)]
            return ClusterNode(centroid, indices, children)

        return HierarchicalClusterIndex(buildNode(np.arange(len(corpus.names)), 0))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.root, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            root = pickle.load(f)
        return HierarchicalClusterIndex(root)

    def labelHardness(self, corpus):
        """
        Attaches a `.hardness` float to every internal ClusterNode (None on
        leaves): the minimum pairwise distance between this node's own
        children's centroids, divided by those children's mean intra-
        cluster spread (mean distance of a child's members to its own
        centroid). LOW hardness means the children sit close together
        relative to how tight each child itself is -- exactly the
        geometric signature of "several similar-but-distinct style
        groups," independent of depth or node size, and computable purely
        from the fixed corpus partition (no model calls). This operationalizes
        geometrically what tree_dip_investigation.py found empirically (the
        model's own per-level accuracy is worst in the middle of the tree,
        where children are neither obviously-different macro-categories nor
        already-near-duplicate leaves): a node whose children are
        geometrically hard to tell apart is a reasonable, cheap PRIOR for
        "the model's conditioning signal will likely be unreliable here
        too," usable to force a real ask independent of whatever the model
        itself reports at decision time.
        """
        from scipy.spatial.distance import pdist

        whitened = corpus.whitenedMatrix.cpu().numpy()

        def visit(node):
            if not node.children:
                node.hardness = None
                return
            centroids = np.stack([c.centroid for c in node.children])
            centroidDists = pdist(centroids)
            minCentroidDist = centroidDists.min() if centroidDists.size else 0.0
            spreads = []
            for c in node.children:
                members = whitened[c.memberIndices]
                spreads.append(np.linalg.norm(members - c.centroid, axis=1).mean())
            meanSpread = float(np.mean(spreads)) + 1e-6
            node.hardness = float(minCentroidDist) / meanSpread
            for c in node.children:
                visit(c)

        visit(self.root)

    def hardNodeIds(self, percentile=30):
        """
        Returns a set of id(node) for every internal node whose `.hardness`
        (see labelHardness, must be called first) is at or below the given
        percentile among all internal nodes -- i.e. the geometrically
        hardest fraction of decision points in the whole tree.
        """
        hardnesses = []

        def collect(node):
            if node.children:
                hardnesses.append(node.hardness)
                for c in node.children:
                    collect(c)

        collect(self.root)
        threshold = np.percentile(hardnesses, percentile)

        ids = set()

        def mark(node):
            if node.children:
                if node.hardness <= threshold:
                    ids.add(id(node))
                for c in node.children:
                    mark(c)

        mark(self.root)
        return ids
