"""
A simulated user ("oracle") for the branching text-diffusion search in
branching.py. Since the ground-truth target's ACCEPTANCE SET (the target
font plus its near-duplicates -- see runSession's acceptanceIdx parameter)
is known in simulation (never fed to the diffusion model itself -- only
used to score which branch's preview fonts are closest to it), the oracle
can always pick the objectively best available branch at each decision
point. Injecting a configurable per-decision-point error probability lets
evaluate_branching.py measure how much the whole pipeline's recall@k
actually depends on the user picking correctly -- i.e. how forgiving this
search design is of a real, imperfect human.

A first attempt at making wrong/noisy decisions recoverable (a global
L2-logistic belief refit from each round's shown-vs-chosen fonts, see
belief.py) measured WORSE than doing nothing: ~0-1% recall@10 on its own,
and diluted the working "hard" tree-path ranking when blended with it,
especially at higher noise -- too few, too geographically narrow labels
(a handful of rounds along one converging path) to fit a useful GLOBAL
linear ranking over a 39,421x64 space; that's a very different label
regime than utils/search.py's GridFeedbackSearch, which accumulates many
more rounds over a much more broadly-explored corpus. Reserve/hedge
(below) is the current attempt: cheaper, and structurally closer to what
actually failed (a hard, irreversible per-round elimination), not a
bolted-on separate model.
"""
import numpy as np
import torch

from branching import advanceTo, initParticles, findDecisionPoint, previewFonts, pruneAndResample


def oracleChoice(previewsByCluster, acceptanceIdx, corpus, noiseProb, rng):
    """
    Picks a cluster id from previewsByCluster (cluster id -> [font name]).
    With probability noiseProb, picks a uniformly random cluster instead
    of the true best one -- an independent per-decision-point error
    modeling a user who occasionally judges the wrong group of previewed
    fonts to be headed toward what they want. A cluster's "true" score is
    the closest (smallest whitened-space distance) pairing between ITS
    shown fonts and any font in the acceptance set -- the oracle only
    ever sees what a real user would see, and (like a real user glancing
    at a font) recognizes a close variant of what they want as promising
    too, not only the single exact target.
    """
    clusters = sorted(previewsByCluster.keys())
    if rng.random() < noiseProb:
        return rng.choice(clusters)

    acceptanceVecs = corpus.whitenedMatrix[acceptanceIdx]  # [T, pcaDim]
    best, bestDist = clusters[0], float("inf")
    for cluster in clusters:
        idx = [corpus.nameToIndex[n] for n in previewsByCluster[cluster] if n in corpus.nameToIndex]
        if not idx:
            continue
        dist = torch.cdist(corpus.whitenedMatrix[idx], acceptanceVecs).min().item()
        if dist < bestDist:
            bestDist, best = dist, cluster
    return best


def runSession(diffusion, model, text, acceptanceIdx, corpus, hierarchicalIndex, visualDim,
               numParticles=40, numSteps=50, maxModes=4, topPerCluster=10, numPreviews=6, checkEvery=1,
               minScheduleFraction=0.1, minClusterSize=2, persistFor=4, maxRounds=5, reserveSize=5,
               hedgeMaxBranchSize=500, forceAskThreshold=1000, guidanceScale=1.0, nullText=None,
               noiseProb=0.0, rng=None, device="cpu", assignFn=None, forceAskDepths=frozenset(),
               hardNodeIds=None):
    """
    assignFn: passed through to branching.findDecisionPoint -- swap in
    branching.assignToChildrenEntropy (or any other (x0Hat, node, maxModes,
    minClusterSize, forceMinChildren) -> (labels, numQualifying) callable)
    to change how the "is this actually ambiguous" call is made, independent
    of everything else here. Defaults to assignToChildren (raw count
    threshold) when None.

    forceAskDepths / hardNodeIds: two independent, STATIC ways to force a
    real ask (forceMinChildren=2, same mechanism forceAskThreshold already
    uses for oversized nodes) regardless of what the model's own batch
    currently looks like -- forceAskDepths by tree depth (a prior belief
    that certain depths are unreliable, e.g. the measured depth-2/3 dip),
    hardNodeIds by a precomputed per-node geometric hardness label (see
    corpus.HierarchicalClusterIndex.labelHardness/hardNodeIds -- a prior
    belief tied to the SPECIFIC node's own children's separation, not just
    its depth). All three forcing conditions (size, depth, hard-node) are
    OR'd together with the existing forceAskThreshold check.
    """
    """
    Runs one full interactive session for a single query, oracle-driven
    end to end: at each step either (a) silently auto-descends into a
    single clearly-favored child (no ambiguity -- nothing to ask), (b)
    finds a genuine decision point among 2+ competing children, resolves
    their preview fonts, and lets the (possibly noisy) oracle choose, (c)
    has used up its maxRounds asks but the tree is still ambiguous, in
    which case it silently follows whichever child the model itself
    supports most rather than stalling at the current node, or (d)
    reaches x_0 with the batch settled. Continues until the particle
    batch reaches x_0 or a leaf node is hit. Only case (b) counts against
    maxRounds -- an auto-descend isn't a choice the user had to make.

    Auto-descending on an unambiguous clear favorite matters: earlier
    this only fired on a genuine >=2-way tie, and anything short of that
    (including a single dominant child with no real competitor -- the
    common case) was treated as "nothing to do," leaving sessions stuck
    at high tree levels (measured median final node size ~100) instead of
    actually walking down the tree the way an always-descend oracle does
    on the exact same tree (measured median leaf size ~4 within 5 rounds).

    reserveSize: at every ASKED decision (case (b)), reserveSize already-
    computed preview completions from EACH REJECTED branch (previewFonts
    rolls these out for every shown branch anyway -- reused here, not
    recomputed) are stashed into a reserve pool ("reserveX"), and every
    rejected branch's own real font membership is added to a running
    "candidatePool" set. Why: pruneAndResample hard-commits 100% of the
    particle mass to the chosen branch every round -- a real, structural
    risk when a single round decides over a large fraction of the corpus
    on necessarily imperfect signal (a real user's judgment, or this
    model's own measured ~47% top-1 accuracy at the root). The reserve
    doesn't change which branch gets explored further (only the chosen
    one does -- this isn't a full beam search), but it keeps a cheap,
    honest record of "what the rejected branch's own best guess at x_0
    looked like," so a final ranking that also draws on the reserve can
    still credit a font from a branch that got rejected -- possibly
    wrongly, possibly by a noisy pick -- instead of that branch's
    evidence being discarded outright.

    candidatePool matters as much as reserveX: an earlier version ranked
    reserveX + finalX against the FULL 39,421-font corpus and measured
    WORSE recall than finalX alone -- adding more particles to a best-of-N
    ranking helps distractor candidates just as readily as the true one,
    and there are vastly more possible distractors than true positives,
    so the net effect was dilution, not recovery (the same effect
    measured separately: scaling up single-shot sample count alone hurt
    recall for the same reason). Restricting the hedge's final ranking to
    ONLY the branches actually considered (candidatePool) was the first
    fix, but still measured a candidatePool averaging ~19,000 fonts (half
    the corpus!) -- because EARLY rounds reject branches that are
    themselves huge (a root-level rejection covers ~4,000 fonts), so
    "every rejected branch, unconditionally" reintroduces the same
    dilution one level down. hedgeMaxBranchSize is the actual fix: a
    rejected branch only joins the hedge if its own membership is small
    enough (<= hedgeMaxBranchSize) that reconsidering it is cheap and its
    reserve completions (a handful of particles) can meaningfully re-rank
    it. This matches the real distinction the user pointed at: an early,
    large-scale decision genuinely needs to be trusted to whatever judged
    it (the oracle, or a real user) -- there's no cheap way to hedge
    against being wrong about which quarter of the corpus to explore. A
    late, small-scale decision is exactly where hedging is cheap AND
    where dilution from admitting a huge extra candidate set is avoided.

    forceAskThreshold: nodes with more members than this NEVER auto-
    descend, even when only one child clearly qualifies -- the runner-up
    (by raw particle count, whatever it is) gets padded in as a second
    candidate and the oracle is asked to confirm/override. The model's
    own unsupervised top pick isn't any more trustworthy at a large node
    than a small one (root-level top-1 accuracy measured ~47%), but being
    wrong about it is far more costly there, so it's worth spending an
    ask even on a weak runner-up rather than trusting an unsupervised
    guess over a huge fraction of the corpus.

    acceptanceIdx: corpus indices of the target font's own acceptance set
    (e.g. its top-T nearest neighbors, matching utils/search.py's
    GridFeedbackSearch simulated-user convention per CLAUDE.md), not just
    the single exact font.

    hierarchicalIndex: corpus.HierarchicalClusterIndex -- the fixed corpus
    tree decision points are detected and navigated against.

    Returns {"finalX": [numParticles, visualDim] (fully denoised, all in
    the branch the session ultimately settled in), "reserveX": [<=
    reserveSize * numRejectedBranches * rounds, visualDim] rejected-branch
    completions (may be empty), "candidatePool": set of corpus indices
    (finalNode's members plus every rejected branch's members, across all
    rounds) hedgeRank restricts its ranking to, "rounds": number of ASKED
    decision points the session needed, "history": per-round diagnostics
    (asked decisions only), "finalNode": the corpus.ClusterNode the
    session ended in}.
    """
    rng = rng or np.random.default_rng()
    batch = initParticles(diffusion, text, visualDim, numParticles, numSteps, device,
                           guidanceScale=guidanceScale, nullText=nullText)
    node = hierarchicalIndex.root
    depth = 0
    history = []
    reserveX = []
    candidatePool = set()

    while True:
        if not node.children:
            advanceTo(diffusion, model, batch, 0)
            break

        forceMinChildren = 2 if (len(node.memberIndices) > forceAskThreshold
                                  or depth in forceAskDepths
                                  or (hardNodeIds is not None and id(node) in hardNodeIds)) else 0
        labels, numQualifying = findDecisionPoint(
            diffusion, model, batch, node, checkEvery=checkEvery,
            minScheduleFraction=minScheduleFraction, maxModes=maxModes, minClusterSize=minClusterSize,
            persistFor=persistFor, forceMinChildren=forceMinChildren, assignFn=assignFn)

        if numQualifying == 0:
            break  # reached x_0 without ever stably favoring anything

        atFinalStep = batch.scheduleIndex == 0

        if numQualifying >= 2 and len(history) < maxRounds:
            previews, rawCompletions = previewFonts(diffusion, model, batch, labels, node, corpus,
                                                      topPerCluster=topPerCluster, numPreviews=numPreviews, rng=rng)
            chosen = oracleChoice(previews, acceptanceIdx, corpus, noiseProb, rng)
            history.append({"scheduleIndex": batch.scheduleIndex, "numQualifying": numQualifying,
                             "chosen": chosen, "previews": previews, "nodeSize": len(node.memberIndices)})

            for cluster in rawCompletions:
                if cluster == chosen:
                    continue
                rejectedMembers = node.children[cluster].memberIndices
                if len(rejectedMembers) > hedgeMaxBranchSize:
                    continue  # too large a branch to cheaply hedge on -- trust whatever judged it
                candidatePool.update(rejectedMembers.tolist())
                if reserveSize > 0:
                    reserveX.append(rawCompletions[cluster][:reserveSize])
        else:
            # unambiguous (numQualifying==1), OR still ambiguous but out of asks -- either way, silently
            # follow the model's own most-supported child rather than stalling at the current node.
            nonNegative = labels[labels != -1]
            chosen = int(np.argmax(np.bincount(nonNegative)))

        pruneAndResample(batch, labels, chosen, targetSize=numParticles, rng=rng)
        node = node.children[chosen]
        depth += 1
        if atFinalStep:
            break

    candidatePool.update(node.memberIndices.tolist())
    reserveTensor = torch.cat(reserveX, dim=0) if reserveX else batch.x[:0]
    return {"finalX": batch.x, "reserveX": reserveTensor, "rounds": len(history),
            "history": history, "finalNode": node, "candidatePool": candidatePool}
