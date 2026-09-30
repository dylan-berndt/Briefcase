"""Suggested tags for query words that match nothing: a plain word-vector synonym check.

The words the search does know (single-word aliases of the reviewed vocabulary and the caption-table words) each point
at tag groups. An unknown query word is compared, by cosine similarity of spaCy's word vectors, with those known words;
the tag groups of the nearest ones above a floor are offered as suggestions. Nothing here affects the ranking: the user
decides whether to add a suggested tag. Word vectors put antonyms close together (wet ~ dry); that is accepted.

Needs spaCy and a vectors model (en_core_web_md by default). Without them, suggestions are simply off.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


class SynonymSuggester:
    def __init__(self, vocabulary, groups, model="en_core_web_md", minSimilarity=0.55, topK=5, neighbours=20):
        """vocabulary: the TagVocabulary the index parses with; groups: the tag groups the index can score."""
        import spacy
        self.nlp = spacy.load(model, exclude=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "senter",
                                               "tok2vec"])
        self.minSimilarity, self.topK, self.neighbours = minSimilarity, topK, neighbours

        known = {}
        for key, targets in vocabulary.aliases.items():
            if len(key) == 1 and key[0].isalpha():
                known.setdefault(key[0], set()).update(c for c, w in targets if w >= 0.8)
        for word, wordGroups in vocabulary.wordTags.items():
            known.setdefault(word, set()).update(wordGroups[:2])
        self.words, self.groupsOf, vectors = [], [], []
        for word in sorted(known):
            usable = sorted(g for g in known[word] if g in groups)
            if usable and self.nlp.vocab.has_vector(word):
                self.words.append(word)
                self.groupsOf.append(usable)
                vectors.append(self.nlp.vocab.get_vector(word))
        matrix = np.stack(vectors).astype(np.float32)
        self.matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-9)

    def suggest(self, word, exclude=()):
        """[(group, via word, similarity)], best first; [] for a word without a vector or with nothing close."""
        if not self.nlp.vocab.has_vector(word):
            return []
        vector = self.nlp.vocab.get_vector(word).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return []
        similarity = self.matrix @ (vector / norm)
        out, seen = [], set(exclude)
        for i in np.argsort(-similarity)[:self.neighbours]:
            if similarity[i] < self.minSimilarity:
                break
            if self.words[i] == word:
                continue
            for group in self.groupsOf[i]:
                if group not in seen:
                    seen.add(group)
                    out.append((group, self.words[i], round(float(similarity[i]), 3)))
        return out[:self.topK]


def loadSuggester(vocabulary, groups, model):
    """The suggester, or None (logged) when spaCy or the vectors model is not installed."""
    if not model:
        return None
    try:
        return SynonymSuggester(vocabulary, groups, model)
    except (ImportError, OSError) as error:
        log.warning("synonym suggestions are off: %s", error)
        return None
