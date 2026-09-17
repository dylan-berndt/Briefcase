"""
A soft, corpus-wide belief accumulator for the branching search -- an
alternative to scoring the final answer by which hierarchy node the
session happened to end in.

Why this exists: branching.pruneAndResample hard-prunes particles down to
literally one hierarchy child's member list every round. That makes every
decision -- auto-descend OR asked -- an IRREVERSIBLE elimination of
everything outside that subtree. That's a real, structural risk: an
early round decides over a huge fraction of the corpus (round 1 of a
branching-10 tree eliminates ~90% of it) based on necessarily imperfect
signal (a real user's judgment, or this project's own measured ~47%
top-1 model accuracy at the root) -- get it wrong once and the true
target is permanently unreachable for the rest of the session, no matter
how much better evidence arrives later. Large, high-stakes, arbitrary
(k-means-boundary, not necessarily style-meaningful) eliminations are
exactly the kind of decision that should NOT be made irreversibly on
noisy input.

utils/search.py's GridFeedbackSearch already solves this differently:
instead of ever partitioning/eliminating the corpus, it refits an
L2-regularized logistic-regression belief over ALL accumulated labels
from scratch every round (its own _fitLogistic). A single noisy or wrong
label just contributes one regularized data point to an aggregate fit --
it nudges the belief, it doesn't permanently delete 90% of the corpus.

BeliefAccumulator applies the same idea here: each round's decision
(chosen branch's preview fonts vs. the OTHER shown branches' preview
fonts) contributes positive/negative labeled examples in the corpus's
whitened space. The final score for every font in the corpus is the
fitted weight's dot product with its own whitened vector -- not
membership in whatever hierarchy node the session happened to end in.
"""
import numpy as np
from scipy.optimize import minimize


def fitLogistic(Z, y, regularization=1.0, iterations=200):
    """
    L2-regularized logistic regression, refit from scratch every call:
        argmin_w  sum_j log(1 + exp(-y_j w.z_j)) + (lambda/2)||w||^2
    Z: [n, dim] feature rows (numpy). y: [n] labels in {-1, +1}.
    Same formulation as utils/search.py's GridFeedbackSearch._fitLogistic,
    reimplemented locally to avoid coupling this experimental package to
    that module's unrelated (transformers/ViT) import weight.
    """
    def lossAndGrad(w):
        margins = y * (Z @ w)
        loss = np.logaddexp(0, -margins).sum() + 0.5 * regularization * (w @ w)
        p = 1.0 / (1.0 + np.exp(margins))
        grad = -(y * p) @ Z + regularization * w
        return loss, grad

    result = minimize(lossAndGrad, np.zeros(Z.shape[1]), jac=True,
                       method="L-BFGS-B", options={"maxiter": iterations})
    return result.x


class BeliefAccumulator:
    """
    Grows a labeled (Z, y) set round by round and refits from scratch
    each time (cheap at D~64 and at most a few dozen labels per session).
    """

    def __init__(self, regularization=1.0):
        self.regularization = regularization
        self.rows = []
        self.labels = []
        self.weight = None

    def addRound(self, positiveVecs, negativeVecs):
        """positiveVecs/negativeVecs: [k, dim] numpy arrays of this round's labeled examples."""
        for v in positiveVecs:
            self.rows.append(v)
            self.labels.append(1.0)
        for v in negativeVecs:
            self.rows.append(v)
            self.labels.append(-1.0)
        self._refit()

    def _refit(self):
        y = np.array(self.labels)
        if len(np.unique(y)) < 2:
            positives = [r for r, label in zip(self.rows, self.labels) if label == 1]
            self.weight = np.mean(positives, axis=0) if positives else None
            return
        Z = np.stack(self.rows, axis=0)
        self.weight = fitLogistic(Z, y, self.regularization)

    def score(self, candidateMatrix):
        """candidateMatrix: [N, dim] numpy. Returns [N] scores, higher = more likely correct."""
        if self.weight is None:
            return np.zeros(candidateMatrix.shape[0])
        return candidateMatrix @ self.weight
