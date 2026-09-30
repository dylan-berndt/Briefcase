"""Free-text query -> ranked fonts, straight off the tagger's scores.

The query is parsed into weighted tag groups by utils/tagVocabulary.py (aliases, negation). Each font is then scored
with a semantic multinomial (Turnbull et al. 2008): its tag probabilities are normalized to sum to 1 over the whole
vocabulary, which removes the bias towards fonts that score high on every tag. A positive group scores
w * log(mass the font puts on the group's tags); a negated group scores |w| * log(probability the font has none of
them), which is bounded, unlike the mirror image of the positive term. The scores are summed.

Words the vocabulary does not know can be mapped to tags learned from caption co-occurrence (configs/wordTags.json,
built by site/tools/buildWordTags.py). searchDetailed() uses them: each such word becomes one group over the union of
its tags ("any of these") at INFERRED_WEIGHT, and is reported separately so it can be shown and removed.
A word that still matches nothing gets suggested tags from a word-vector synonym check (synonyms.py); those do not
affect the ranking until the user adds them (tags=).
"""

import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EPS = 1e-12
INFERRED_WEIGHT = 0.5


def loadVocabularyClass():
    # In the image the parser is copied next to the app; in the repo it lives in utils/, which is a torch package
    # that must not be imported here, so it is loaded by path.
    try:
        from tagVocabulary import TagVocabulary
        return TagVocabulary
    except ImportError:
        path = os.path.join(REPO, "utils", "tagVocabulary.py")
        spec = importlib.util.spec_from_file_location("tagVocabulary", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TagVocabulary


def findVocabularyConfig():
    for directory in (os.path.join(HERE, "configs"), os.path.join(REPO, "configs")):
        path = os.path.join(directory, "tagVocabulary.json")
        if os.path.exists(path):
            return path
    raise FileNotFoundError("configs/tagVocabulary.json not found")


class TagIndex:
    def __init__(self, bundle, vocabularyPath=None, synonymModel=None):
        self.bundle = bundle
        self.logits = bundle.logits  # [numTags, numFonts] float16, memory mapped
        self.numFonts = self.logits.shape[1]
        self.tagRow = {tag: row for row, tag in enumerate(bundle.vocab)}
        self.vocabulary = loadVocabularyClass()(vocabularyPath or findVocabularyConfig(), extraTags=bundle.vocab)

        self.logMass = self.logTotalMass()
        self.groups = {}
        for name, entry in self.vocabulary.canonical.items():
            rows = [self.tagRow[m] for m in entry["members"] if m in self.tagRow]
            if rows:
                self.groups[name] = np.array(sorted(set(rows)))

        from synonyms import loadSuggester
        self.suggester = loadSuggester(self.vocabulary, self.groups, synonymModel)

    def logTotalMass(self, chunk=128):
        mass = np.zeros(self.numFonts, dtype=np.float64)
        for start in range(0, self.logits.shape[0], chunk):
            block = self.logits[start:start + chunk].astype(np.float32)
            mass += (1.0 / (1.0 + np.exp(-block))).sum(axis=0)
        return np.log(mass + EPS).astype(np.float32)

    def parse(self, query):
        """Returns ([(group, weight)], unmatched words). Groups the model has no tags for count as unmatched."""
        weights, unmatched = self.vocabulary.parse(query)
        terms = []
        for name, weight in weights.items():
            if name in self.groups:
                terms.append((name, weight))
            else:
                unmatched.append(name)
        return terms, unmatched

    def parseDetailed(self, query, ignore=()):
        """Like parse(), plus words matched only through the caption table: (terms, unmatched, inferred) where
        inferred is [(word, [groups], weight)]. Words in `ignore` are not inferred (the user removed that guess)."""
        found = {}
        weights, unmatched = self.vocabulary.parse(query, inferred=found)
        terms = []
        for name, weight in weights.items():
            if name in self.groups:
                terms.append((name, weight))
            else:
                unmatched.append(name)
        inferred = []
        for word, (sign, groups) in found.items():
            groups = [g for g in groups if g in self.groups]
            if word in ignore or not groups:
                unmatched.append(word)
            else:
                inferred.append((word, groups, sign * INFERRED_WEIGHT))
        return terms, unmatched, inferred

    def score(self, terms, inferred=()):
        """terms: [(group, weight)], inferred: [(word, [groups], weight)] -> float32 [numFonts], higher is better."""
        total = np.zeros(self.numFonts, dtype=np.float32)
        blocks = [(self.groups[name], weight) for name, weight in terms]
        blocks += [(np.array(sorted(set(np.concatenate([self.groups[g] for g in groups])))), weight)
                   for _, groups, weight in inferred]
        for rows, weight in blocks:
            block = self.logits[rows].astype(np.float32)  # [members, numFonts]
            if weight > 0:
                mass = (1.0 / (1.0 + np.exp(-block))).sum(axis=0)
                total += weight * (np.log(mass + EPS) - self.logMass)
            else:
                # log prod(1 - p) = -sum softplus(logit)
                total += weight * np.logaddexp(0.0, block).sum(axis=0)
        return total

    def search(self, query):
        """Full ranking of the corpus: (font indices best first, terms, unmatched). Empty when nothing matched."""
        terms, unmatched = self.parse(query)
        if not terms:
            return np.empty(0, dtype=np.int64), terms, unmatched
        # stable, so equal scores keep corpus order and pages never overlap or skip
        order = np.argsort(-self.score(terms), kind="stable")
        return order, terms, unmatched

    def searchDetailed(self, query, ignore=(), tags=()):
        """search() with caption-inferred tags, tags the user added and synonym suggestions:
        (order, terms, unmatched, inferred, suggested). tags: [(group, weight)] added on top of the query (a name the
        index does not know goes to unmatched); suggested: [(word, [(group, via word, similarity)])] for words that
        matched nothing, not used in the ranking."""
        terms, unmatched, inferred = self.parseDetailed(query, ignore)
        present = {name for name, _ in terms}
        for name, weight in tags:
            if name not in self.groups:
                unmatched.append(name)
            elif name not in present:
                terms.append((name, weight))
                present.add(name)
        suggested = []
        if self.suggester is not None:
            taken = present | {g for _, groups, _ in inferred for g in groups}
            for word in unmatched:
                if word.isalpha() and word not in self.groups:
                    options = self.suggester.suggest(word, exclude=taken)
                    if options:
                        suggested.append((word, options))
        if not terms and not inferred:
            return np.empty(0, dtype=np.int64), terms, unmatched, inferred, suggested
        order = np.argsort(-self.score(terms, inferred), kind="stable")
        return order, terms, unmatched, inferred, suggested
