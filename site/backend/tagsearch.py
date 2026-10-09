"""Free-text query -> ranked fonts, straight off the tagger's scores.

The query is parsed into weighted tag groups by utils/tagVocabulary.py (aliases). Each font is then scored with a
semantic multinomial (Turnbull et al. 2008): its tag probabilities are normalized to sum to 1 over the whole
vocabulary, which removes the bias towards fonts that score high on every tag. A group scores
w * log(mass the font puts on the group's tags); the scores are summed. There is no negation: a "not x" in a query
yields no tag for x.

Words the vocabulary does not know can be mapped to tags learned from caption co-occurrence (configs/wordTags.json,
built by site/tools/buildWordTags.py); describe() turns each such guess into ordinary tags at INFERRED_WEIGHT. A word
that still matches nothing gets suggested tags from a WordNet + Datamuse table (synonyms.py). describe() answers "which
tags does this query mean"; rank() orders the fonts for exactly the tags it is given, so the page can keep the user's
ticks itself and send the final list.
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
    def __init__(self, bundle, vocabularyPath=None, synonymTable=""):
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

        # every group's member rows back to back, so one maximum.reduceat gives all group probabilities at once
        self.groupNames = list(self.groups)
        self.memberRows = np.concatenate([self.groups[name] for name in self.groupNames]) if self.groups else \
            np.empty(0, dtype=np.int64)
        self.memberStarts = np.cumsum([0] + [len(self.groups[name]) for name in self.groupNames[:-1]])

        from synonyms import loadSuggester
        self.suggester = loadSuggester(self.groups, synonymTable, self.vocabulary.stem)

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

    def parseDetailed(self, query):
        """Like parse(), plus words matched only through the caption table: (terms, unmatched, inferred) where
        inferred is [(word, [groups], weight)]."""
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
            if not groups:
                unmatched.append(word)
            else:
                inferred.append((word, groups, sign * INFERRED_WEIGHT))
        return terms, unmatched, inferred

    def score(self, terms):
        """terms: [(group, weight)] -> float32 [numFonts], higher is a better match."""
        total = np.zeros(self.numFonts, dtype=np.float32)
        for name, weight in terms:
            block = self.logits[self.groups[name]].astype(np.float32)  # [members, numFonts]
            mass = (1.0 / (1.0 + np.exp(-block))).sum(axis=0)
            total += weight * (np.log(mass + EPS) - self.logMass)
        return total

    def rank(self, terms):
        """Full ranking of the corpus for [(group, weight)]: font indices, best first. Empty without terms."""
        if not terms:
            return np.empty(0, dtype=np.int64)
        # stable, so equal scores keep corpus order and pages never overlap or skip
        return np.argsort(-self.score(terms), kind="stable")

    def groupProbabilities(self, fonts):
        """[numGroups, len(fonts)] float32, rows in groupNames order: the probability that each font has each group,
        taken from its most confident member tag (members are spelling variants of one tag)."""
        block = np.asarray(self.logits[:, np.sort(fonts)], dtype=np.float32)[self.memberRows]
        return np.maximum.reduceat(1.0 / (1.0 + np.exp(-block)), self.memberStarts, axis=0)

    def refinements(self, terms, top=200, count=8, redundancy=0.7, minVariance=0.01):
        """Tags that would best split the current top results, to offer as the next tag to add.

        Each group's probability is taken over the `top` best fonts for terms; groups are ranked by its variance,
        which is highest for a tag that about half of those fonts confidently have and half confidently lack (not a
        rare or a common one, and not one the model is unsure about). Going down that ranking, a group whose
        correlation with one already listed (offered or opposite) is above `redundancy` splits the fonts the same way
        and is skipped; one below -redundancy against an offered tag splits them the opposite way (wide against
        narrow) and is listed as that tag's opposite. Groups already in terms are never offered.

        Returns [(group, share, opposites)]: share is the mean probability over the top fonts, opposites is
        [(group, share)]."""
        order = self.rank(terms)[:top]
        if len(order) < 2:
            return []
        probs = self.groupProbabilities(order)
        share = probs.mean(axis=1)
        centred = probs - share[:, None]
        variance = (centred ** 2).mean(axis=1)
        norms = np.sqrt((centred ** 2).sum(axis=1))
        present = {name for name, _ in terms}

        def correlation(a, b):
            return centred[a] @ centred[b] / (norms[a] * norms[b])

        chosen, opposites, listed = [], {}, []
        for g in np.argsort(-variance, kind="stable"):
            if len(chosen) == count or variance[g] < minVariance:
                break
            if self.groupNames[g] in present:
                continue
            # anything already listed, offered or opposite, that splits the same way makes this one a repeat
            if any(correlation(g, other) > redundancy for other in listed):
                continue
            listed.append(g)
            againstChosen = [correlation(g, c) for c in chosen]
            if againstChosen and min(againstChosen) < -redundancy:
                opposites[chosen[int(np.argmin(againstChosen))]].append(g)
                continue
            chosen.append(g)
            opposites[g] = []
        return [(self.groupNames[g], float(share[g]), [(self.groupNames[o], float(share[o])) for o in opposites[g]])
                for g in chosen]

    def describe(self, query, suggest=True):
        """What a query means as a list of tags: (terms, suggested, unmatched).

        terms: [(group, weight)], the tags the query matched (a negated one, "not x", is dropped), plus the tags
        guessed for words the vocabulary does not know, each at INFERRED_WEIGHT. suggested: [(group, via word,
        score)], WordNet/Datamuse-related tags for words that matched nothing; they are not part of terms, the caller decides whether
        to use them. unmatched: words that matched nothing and got no suggestion."""
        terms, unmatched, inferred = self.parseDetailed(query)
        terms = [(name, weight) for name, weight in terms if weight > 0]
        present = {name for name, _ in terms}
        for _, groups, weight in inferred:
            if weight <= 0:
                continue
            for group in groups:
                if group not in present:
                    terms.append((group, weight))
                    present.add(group)
        suggested, left, seen = [], [], set(present)
        for word in unmatched:
            options = []
            if suggest and self.suggester is not None and word.isalpha() and word not in self.groups:
                options = [o for o in self.suggester.suggest(word, exclude=seen) if o[0] not in seen]
            for group, via, score in options:
                seen.add(group)
                suggested.append((group, via, score))
            if not options:
                left.append(word)
        return terms, suggested, left

    def search(self, query):
        """rank() for what the query means: (font indices best first, terms, unmatched)."""
        terms, _, unmatched = self.describe(query, suggest=False)
        return self.rank(terms), terms, unmatched

    def parseChoices(self, text, limit=64):
        """The final tag list a page sends back: "name", each optionally ":weight" (0-1, the weight describe()
        reported). Names the index does not know are skipped. Raises ValueError for a malformed entry."""
        terms = {}
        entries = [e.strip().lower() for e in text.split(",") if e.strip()]
        if len(entries) > limit:
            raise ValueError(f"at most {limit} tags")
        for entry in entries:
            if entry.startswith("-"):
                raise ValueError(f"{entry!r}: tags cannot be excluded")
            name, _, weight = entry.partition(":")
            try:
                weight = float(weight) if weight else 1.0
            except ValueError:
                raise ValueError(f"bad weight in {entry!r}")
            if not 0 < weight <= 1:
                raise ValueError(f"weight of {name!r} must be above 0 and at most 1")
            if name in self.groups:
                terms[name] = weight
        return list(terms.items())
