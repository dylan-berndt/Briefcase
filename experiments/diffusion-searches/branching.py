"""
Portable primitives for the interactive "decision point" font search: a
shared reverse-diffusion trajectory is run as a batch of particles (all
conditioned on one text query), paused whenever the particles' trajectory
is clearly heading toward more than one child of the CURRENT node in a
precomputed hierarchical corpus partition (corpus.HierarchicalClusterIndex),
narrowed to whichever child a user (or, in oracle.py, a simulated one)
picks, and resumed one level deeper -- so the user is only asked to
choose again when the model's own output has actually become ambiguous,
instead of once per fixed round (GridFeedbackSearch's approach) or not at
all (a single evaluate.py-style sample).

Every function here is pure torch/numpy with no CLI or file I/O, and
operates on one ParticleBatch at a time -- the same call sequence
(advanceTo / findDecisionPoint / previewFonts / pruneAndResample), stepping
down the SAME hierarchy tree one node at a time, that oracle.py drives
with a simulated user is what a web endpoint would drive one HTTP request
at a time with a real one, holding a ParticleBatch's state (x, x0Hat,
text, steps, scheduleIndex) and the current tree node between requests.

Two real design pivots got here, both from direct measurement:

1. Decision points are detected against a FIXED, precomputed partition of
   the real font corpus, not by directly (re-)clustering the small live
   particle batch every step. Live-clustering was tried first and didn't
   hold up: bisecting a 30-particle batch with agglomerative (Ward)
   clustering found real-query "splits" statistically indistinguishable
   from splits found doing the same thing to pure, unconditioned noise
   (measured: real-query max gap ratio 1.17-1.50 across 15 queries;
   random-text null baseline over 10 trials: mean 1.21, std 0.09, max
   1.47). Testing each particle's cheap one-step x0 estimate
   (diffusion.GaussianDiffusion.reverseSteps' x0Hat -- already computed
   for free) against fixed centroids instead turns "does this noisy
   little sample look clustered" (no signal) into "is the model
   consistently favoring one real corpus region over another" (an
   actual, checkable claim), and is a cheap nearest-centroid lookup
   rather than a per-step clustering call.
2. That fixed partition was then made HIERARCHICAL, not flat. A single
   flat k-way partition, reused every round, caps addressable resolution
   at k regardless of round count -- once particles settle into one
   region, re-checking against the SAME global centroids doesn't narrow
   any further within it. Measured effect: a flat 16-region version of
   this search got recall@10 ~10%. Navigating a depth-5, branching-10
   hierarchical tree of the SAME corpus with an OMNISCIENT oracle (no
   diffusion model involved, just always stepping into the child that
   truly contains a known target) got a median final leaf of 4 fonts in
   <=5 rounds -- implied recall@10 ~97%. The flat partition, not the
   embedding space's information content, was the bottleneck.
"""
import numpy as np
import torch


class ParticleBatch:
    """
    x: [P, visualDim] current state, at the noise level associated with
    scheduleIndex, for P particles sharing one text conditioning vector
    (text is stored per-particle, already broadcast, purely so pruning/
    resampling can index it the same way as x).

    x0Hat: [P, visualDim] the one-step x0 estimate from the last update
    applied (see diffusion.GaussianDiffusion.reverseSteps) -- None before
    the first advanceTo call. This is what findDecisionPoint tests
    against ClusterIndex; it is NOT a valid final answer on its own
    (blurrier than a completed sample, see reverseSteps' docstring).

    scheduleIndex follows diffusion.reverseSteps' convention: it names the
    schedule position whose update has already been applied, so x at
    scheduleIndex==0 is the fully denoised x_0, and x at
    scheduleIndex==len(steps)-1 is x_T (fresh noise, no update applied yet).
    """

    def __init__(self, x, text, steps, scheduleIndex, x0Hat=None, guidanceScale=1.0, nullText=None):
        self.x = x
        self.text = text
        self.steps = steps
        self.scheduleIndex = scheduleIndex
        self.x0Hat = x0Hat
        self.guidanceScale = guidanceScale
        self.nullText = nullText  # [P, textDim] or None, broadcast the same way as text

    @property
    def numParticles(self):
        return self.x.shape[0]


def initParticles(diffusion, text, visualDim, numParticles, numSteps, device, guidanceScale=1.0, nullText=None):
    """
    text: [textDim] single query embedding, broadcast across particles.
    guidanceScale/nullText: classifier-free guidance -- see diffusion.
    GaussianDiffusion.reverseSteps' docstring for what this buys (measured
    ~1.5x improvement in own-target precision at scale 2-4 on this
    checkpoint) and why it's needed at all (a full-corpus coverage
    experiment found reachability is NOT the bottleneck -- ~96%+ of the
    corpus is approximated within 2x real-neighbor tightness by SOME
    sample already -- PRECISION to a SPECIFIC query's own target is).
    nullText, if given: [textDim] single pseudo-unconditional embedding
    (e.g. the corpus-wide mean query embedding), broadcast the same way.
    """
    steps = diffusion.respacedSteps(numSteps)
    textBatch = text.unsqueeze(0).expand(numParticles, -1).contiguous().to(device)
    nullTextBatch = nullText.unsqueeze(0).expand(numParticles, -1).contiguous().to(device) if nullText is not None else None
    x = torch.randn(numParticles, visualDim, device=device)
    return ParticleBatch(x, textBatch, steps, scheduleIndex=len(steps) - 1,
                          guidanceScale=guidanceScale, nullText=nullTextBatch)


def advanceTo(diffusion, model, batch, targetIndex):
    """Advances batch in place from its current scheduleIndex down to targetIndex. No-op if already there."""
    if batch.scheduleIndex <= targetIndex:
        return batch
    x, x0Hat = batch.x, batch.x0Hat
    for _, x, x0Hat in diffusion.reverseSteps(model, batch.text, batch.x.shape[-1], device=batch.x.device,
                                               x=batch.x, steps=batch.steps,
                                               startIndex=batch.scheduleIndex, stopIndex=targetIndex,
                                               guidanceScale=batch.guidanceScale, nullText=batch.nullText):
        pass
    batch.x, batch.x0Hat = x, x0Hat
    batch.scheduleIndex = targetIndex
    return batch


def assignToChildren(x0Hat, node, maxModes=4, minClusterSize=3, forceMinChildren=0):
    """
    Assigns each particle's x0Hat to the nearest CHILD of node (a
    corpus.ClusterNode) -- x0Hat already lives in the same whitened space
    the tree was built in, so this is a direct nearest-centroid lookup
    among just node's own children, no extra transform needed -- then
    keeps only the (up to maxModes) most populous children with
    >= minClusterSize particles as real candidates.

    Returns (labels [P] int -- an index into node.children, or -1 for
    particles in a child that didn't qualify, numQualifying):
      numQualifying == 0: no child reached minClusterSize -- inconclusive,
        keep advancing.
      numQualifying == 1: exactly one child clearly has enough support and
        no real competitor -- UNAMBIGUOUS, the trajectory should silently
        descend into it without asking the user (this case used to be
        conflated with numQualifying==0 and treated as "nothing to do",
        which left real sessions stuck at the root: measured median final
        node size ~100 instead of the ~4 an always-descend oracle reaches
        in the same tree -- see corpus.py's docstring).
      numQualifying >= 2: a genuine decision point -- multiple children
        each have real, competing support; the caller should ask the user.

    forceMinChildren pads the kept set up to this many children (by raw
    particle count, even below minClusterSize) whenever at least one
    child already qualifies -- for callers that want to force a genuine
    ask instead of an auto-descend at particularly high-stakes nodes. A
    large node's unsupervised top pick isn't any more trustworthy than a
    small node's (root-level top-1 measured only ~47% accurate), but
    being wrong about it is far more costly, so it's worth spending an
    ask even on a runner-up that wouldn't independently clear
    minClusterSize. Never fires when NO child reaches minClusterSize at
    all (numQualifying stays 0 either way).
    """
    if not node.children:
        return np.full(x0Hat.shape[0], -1, dtype=int), 0

    centroids = torch.from_numpy(np.stack([c.centroid for c in node.children]).astype(np.float32)).to(x0Hat.device)
    assignments = torch.cdist(x0Hat, centroids).argmin(dim=1).cpu().numpy()
    counts = np.bincount(assignments, minlength=len(node.children))
    qualifying = np.where(counts >= minClusterSize)[0]

    if qualifying.size == 0:
        return np.full(x0Hat.shape[0], -1, dtype=int), 0

    keepCount = max(maxModes, 1)
    if forceMinChildren > qualifying.size:
        keepCount = max(keepCount, min(forceMinChildren, len(node.children)))
        order = np.argsort(-counts)
        kept = order[:keepCount]
        kept = kept[counts[kept] > 0]
    else:
        kept = qualifying[np.argsort(-counts[qualifying])][:keepCount]
    labels = np.where(np.isin(assignments, kept), assignments, -1)
    return labels, len(kept)


def assignToChildrenEntropy(x0Hat, node, maxModes=4, minClusterSize=3, forceMinChildren=0, entropyThreshold=0.6):
    """
    Alternative to assignToChildren's raw-count-threshold rule: instead of
    asking "did >= minClusterSize particles land in >= 2 children,"
    computes the normalized Shannon entropy of the particle-count
    distribution across ALL of node's children (0 = every particle in one
    child, 1 = perfectly uniform across all children) and treats the
    distribution as a genuine decision point whenever that entropy clears
    entropyThreshold. This is a different notion of "uncertainty" than a
    hard count threshold: minClusterSize asks "do at least two children
    have enough raw support," which a batch split 36/2/2/0 (out of 40)
    can satisfy despite being overwhelmingly one-sided; entropy asks "how
    UNCERTAIN is the batch's overall vote," which the same 36/2/2/0 split
    scores as low-entropy (confidently one child) even though it
    technically clears a minClusterSize>=2 bar. Same forceMinChildren
    override as assignToChildren (an external caller can still force a
    real ask regardless of the batch's own vote).
    """
    if not node.children:
        return np.full(x0Hat.shape[0], -1, dtype=int), 0

    centroids = torch.from_numpy(np.stack([c.centroid for c in node.children]).astype(np.float32)).to(x0Hat.device)
    assignments = torch.cdist(x0Hat, centroids).argmin(dim=1).cpu().numpy()
    counts = np.bincount(assignments, minlength=len(node.children))
    total = counts.sum()
    if total == 0:
        return np.full(x0Hat.shape[0], -1, dtype=int), 0

    probs = counts[counts > 0] / total
    entropy = -(probs * np.log(probs)).sum() / np.log(len(node.children)) if len(node.children) > 1 else 0.0
    activeChildren = np.where(counts > 0)[0]

    if forceMinChildren >= 2 or entropy >= entropyThreshold:
        keepCount = max(maxModes, 1)
        if forceMinChildren > 0:
            keepCount = max(keepCount, min(forceMinChildren, len(node.children)))
        kept = activeChildren[np.argsort(-counts[activeChildren])][:keepCount]
        if len(kept) < 2:
            if len(kept) == 0:
                return np.full(x0Hat.shape[0], -1, dtype=int), 0
            return np.where(assignments == kept[0], kept[0], -1), 1
        labels = np.where(np.isin(assignments, kept), assignments, -1)
        return labels, len(kept)

    best = int(counts.argmax())
    return np.where(assignments == best, best, -1), 1


def findDecisionPoint(diffusion, model, batch, node, checkEvery=1, minScheduleFraction=0.1,
                       maxModes=4, minClusterSize=3, persistFor=3, forceMinChildren=0, assignFn=None):
    """
    Advances batch step by step toward x_0, checking every checkEvery
    schedule steps which CHILD of node (assignToChildren) each particle's
    current x0Hat estimate is nearest to. Stops and returns as soon as
    the SAME qualifying-child set (whether size 1 -- an unambiguous
    auto-descend -- or >=2 -- a real decision) persists for persistFor
    consecutive checks in a row (batch is left paused there), or once
    batch reaches x_0 without ever qualifying anything stably.

    persistFor matters: a single instant's result is not enough evidence
    on its own -- consecutive x0Hat estimates are highly correlated (each
    is a small step from the last), so a transient, noise-driven
    disagreement at a region boundary tends to look like a "split" for
    one check and then dissolve, and measured false-positive rates for
    a SINGLE uniformly-random check are a poor guide to how often that
    happens in a real, autocorrelated trajectory (calibrating minClusterSize
    against fresh-random-draw false-positive rate alone was tried first
    here and undershot badly -- every real session branched dozens of
    times). Requiring the same qualifying set to reappear several checks
    running is a much stronger, more UX-appropriate bar: a genuine fork
    (or a genuine clear favorite) should look stable over a short stretch
    of the trajectory, not flicker for one step.

    minScheduleFraction skips checking during the earliest part of the
    schedule: with alphaBar still near 0 there, x0Hat is a near-pure
    extrapolation from almost no signal, so child assignments there are
    meaningless regardless of text conditioning.

    Returns (labels, numQualifying) as of wherever the batch ended up --
    see assignToChildren for the 0/1/2+ meaning. 0 here specifically means
    "reached x_0 without ever stably qualifying anything."
    """
    assignFn = assignFn or assignToChildren
    total = len(batch.steps)
    skipAbove = total * (1 - minScheduleFraction)
    labels, numQualifying = np.full(batch.numParticles, -1, dtype=int), 0
    streak, streakKey = 0, None

    while batch.scheduleIndex > 0:
        nextIndex = max(batch.scheduleIndex - checkEvery, 0)
        advanceTo(diffusion, model, batch, nextIndex)
        if batch.scheduleIndex > skipAbove:
            continue

        candidateLabels, candidateCount = assignFn(
            batch.x0Hat, node, maxModes=maxModes, minClusterSize=minClusterSize,
            forceMinChildren=forceMinChildren)

        if candidateCount == 0:
            streak, streakKey = 0, None
            continue

        key = tuple(sorted(set(candidateLabels.tolist()) - {-1}))
        streak = streak + 1 if key == streakKey else 1
        streakKey = key
        if streak >= persistFor:
            return candidateLabels, candidateCount

    return labels, numQualifying


def previewFonts(diffusion, model, batch, labels, node, corpus, topPerCluster=10, numPreviews=6, rng=None):
    """
    For each child (of node) present in labels (excluding unassigned -1
    particles), resolves up to topPerCluster distinct real fonts to
    present to the user as that branch's preview -- searched only among
    THAT CHILD's own real member fonts (node.children[c].memberIndices),
    not the whole corpus, so every previewed font is guaranteed to
    actually belong to the branch it's shown under (an earlier flat-
    partition version of this search didn't have real per-branch
    membership to restrict to, and could show fonts unrelated to the
    branch they were attached to).

    "Several diffusion steps ahead of the decision point" is implemented
    as several independent FULL completions to x_0 (fresh noise for each,
    thrown away afterward -- batch itself is never mutated, it stays
    paused at the decision point for the caller to prune/resume): x0Hat
    is a free, cheap, but blurrier estimate good enough for branch
    assignment, not sharp enough to show as a final candidate font (see
    reverseSteps' docstring). When the batch is already at x_0, previews
    use the particles' own final values directly instead of running a
    redundant extra rollout. Everything happens directly in the corpus's
    whitened space (no ambient round trip -- see corpus.py's docstring).

    All clusters' preview seeds are concatenated into ONE batched
    reverseSteps call rather than one call per cluster: a tiny MLP's
    forward cost barely grows with batch size, so this trades a few
    clusters' worth of Python/call overhead for free.

    Returns (fontsByCluster, rawCompletionsByCluster) -- the second dict
    (cluster id -> [numPreviews, visualDim] raw whitened-space completions,
    the same ones used to resolve fontsByCluster) is exposed so a caller
    can reuse REJECTED clusters' completions as a hedge/reserve (e.g.
    oracle.py) instead of throwing them away and re-rolling later.
    """
    rng = rng or np.random.default_rng()
    clusters = sorted(c for c in set(labels.tolist()) if c != -1)

    if batch.scheduleIndex == 0:
        finalByCluster = {}
        for cluster in clusters:
            idxTensor = torch.as_tensor(np.where(labels == cluster)[0], device=batch.x.device, dtype=torch.long)
            finalByCluster[cluster] = batch.x[idxTensor]
    else:
        pickIdxByCluster = {}
        for cluster in clusters:
            memberIdx = np.where(labels == cluster)[0]
            replace = memberIdx.size < numPreviews
            pickIdxByCluster[cluster] = rng.choice(memberIdx, size=numPreviews, replace=replace)

        allPickIdx = np.concatenate([pickIdxByCluster[c] for c in clusters])
        idxTensor = torch.as_tensor(allPickIdx, device=batch.x.device, dtype=torch.long)
        seedX = batch.x[idxTensor].clone()
        seedText = batch.text[idxTensor].clone()
        seedNullText = batch.nullText[idxTensor].clone() if batch.nullText is not None else None

        finalX = seedX
        for _, finalX, _ in diffusion.reverseSteps(model, seedText, seedX.shape[-1], device=seedX.device,
                                                     x=seedX, steps=batch.steps,
                                                     startIndex=batch.scheduleIndex, stopIndex=0,
                                                     guidanceScale=batch.guidanceScale, nullText=seedNullText):
            pass

        finalByCluster, offset = {}, 0
        for cluster in clusters:
            n = len(pickIdxByCluster[cluster])
            finalByCluster[cluster] = finalX[offset:offset + n]
            offset += n

    results = {}
    for cluster, finalX in finalByCluster.items():
        finalX = finalX.detach().to(corpus.device)
        memberIndices = node.children[cluster].memberIndices
        _, nameRows = corpus.nearest(finalX, k=topPerCluster, restrictTo=memberIndices)
        fonts, seen = [], set()
        for row in nameRows:
            for name in row:
                if name not in seen:
                    seen.add(name)
                    fonts.append(name)
        results[int(cluster)] = fonts[:topPerCluster]
    return results, {int(c): x.detach() for c, x in finalByCluster.items()}


def pruneAndResample(batch, labels, chosenCluster, targetSize=None, rng=None):
    """
    Keeps only the particles in chosenCluster, then resamples with
    replacement back up to targetSize (default: batch's current particle
    count) so the next round has a full batch to search for a decision
    point in. Each resampled duplicate independently draws its own fresh
    noise every subsequent step (ancestral sampling), so duplicating
    survivors is a legitimate way to regrow diversity within the chosen
    branch, not a no-op.
    """
    rng = rng or np.random.default_rng()
    targetSize = targetSize or batch.numParticles
    survivorIdx = np.where(labels == chosenCluster)[0]
    resampled = rng.choice(survivorIdx, size=targetSize, replace=True)
    idxTensor = torch.as_tensor(resampled, device=batch.x.device, dtype=torch.long)
    batch.x = batch.x[idxTensor].clone()
    batch.text = batch.text[idxTensor].clone()
    batch.x0Hat = batch.x0Hat[idxTensor].clone() if batch.x0Hat is not None else None
    batch.nullText = batch.nullText[idxTensor].clone() if batch.nullText is not None else None
    return batch
