"""
Sparse tag-presence extraction: builds a fixed-vocabulary signed vector
per query, marking which style tags (the existing retrieval-head
vocabulary, checkpoints/retrieval/latest/ViT tags/vocab.json -- already
computed, no new captions or LLM calls) are mentioned in that query's
text -- by exact lemma match, substring match, AND spaCy word-vector
synonym similarity, not literal substring alone (users -- and the
existing LLM-generated captions -- don't always use the exact vocab
word: "graceful"/"refined" for "elegant", "chunky" for "bold", "old-
fashioned" for "vintage"). This mirrors utils/search.py's TagSearch.
queryWeights/_termMatches exactly (same matching algorithm, same
NEGATIONS handling), reimplemented standalone here rather than importing
utils.search, since that class also loads the ViT backbone/font dataset/
embedding cache -- heavy dependencies this lightweight experiments/
pipeline deliberately avoids elsewhere.

Motivation (per direct diagnosis, not a first-principles guess): a dense
sentence embedding entangles "which attributes are mentioned" with "how
they're phrased," and can't represent "unstated" separately from "stated
as absent." A single caption only mentions a partial, noisy SUBSET of a
font's true style attributes -- different captions of the same font
mention different subsets, which is the dominant measured noise source
(not paraphrase variation). Making tag PRESENCE (now recognized even
under a synonym, not just an exact vocab word) the explicit
representational unit, instead of hoping a dense embedding implicitly
encodes it, directly targets that.
"""
import json
import os

import numpy as np

VOCAB_PATH = os.path.join("checkpoints", "retrieval", "latest", "ViT tags", "vocab.json")

# Near-universal filler words measured directly (build_tag_presence_cache.py's frequency
# check): "font"/"typeface"/"design"/"style"/"type" appear in a large fraction of ALL
# captions regardless of actual style ("a font that is...", "this typeface has..."), so
# they carry no discriminative information -- pure dilution of the conditioning vector.
# "script"/"serif"/etc. are excluded from this list deliberately: also common, but a REAL
# style attribute, not generic filler.
GENERIC_STOPWORDS = {"font", "typeface", "design", "style", "type"}

NEGATIONS = {"not", "no", "non", "without", "less", "least", "never", "anti", "un"}


class TagMatcher:
    def __init__(self, path=VOCAB_PATH, spacyModel="en_core_web_lg", synonymThreshold=0.5, synonymScale=1.0):
        import spacy

        with open(path, encoding="utf-8") as f:
            self.rawVocab = json.load(f)
        self.numTags = len(self.rawVocab)
        self.synonymThreshold = synonymThreshold
        self.synonymScale = synonymScale

        try:
            self.nlp = spacy.load(spacyModel, disable=["parser", "ner"])
        except OSError:
            print(f"spaCy model '{spacyModel}' not found; falling back to en_core_web_sm "
                  f"(synonym matching disabled). Install with: python -m spacy download {spacyModel}")
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        self.validIdx = [i for i, t in enumerate(self.rawVocab)
                          if t.lower().strip() not in GENERIC_STOPWORDS and len(t.strip()) > 2]

        self.lemmaToTags = {}
        vectors = []
        hasVectors = self.nlp.vocab.vectors_length > 0
        for i in self.validIdx:
            tag = self.rawVocab[i]
            doc = self.nlp(tag.lower().replace("_", " ").replace("-", " "))
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                self.lemmaToTags.setdefault(token.lemma_, []).append(i)
            if hasVectors:
                norm = doc.vector_norm
                vectors.append(doc.vector / norm if norm else np.zeros_like(doc.vector))
        self.tagVectors = np.stack(vectors).astype(np.float32) if hasVectors else None
        self.tagVectorIdx = self.validIdx if hasVectors else None

        self._tagLowerByIdx = {i: self.rawVocab[i].lower() for i in self.validIdx}

    def _termMatches(self, token):
        matches = {}
        lemma, text = token.lemma_, token.text
        for i in self.lemmaToTags.get(lemma, []):
            matches[i] = max(matches.get(i, 0.0), 1.0)
        for i in self.validIdx:
            tagLower = self._tagLowerByIdx[i]
            if (text and text in tagLower) or (lemma and lemma in tagLower):
                matches[i] = max(matches.get(i, 0.0), 0.5)
        if matches:
            return matches
        if self.tagVectors is not None and token.has_vector and token.vector_norm:
            sims = self.tagVectors @ (token.vector / token.vector_norm)
            for pos in np.where(sims >= self.synonymThreshold)[0]:
                i = self.tagVectorIdx[pos]
                matches[i] = float(sims[pos]) * self.synonymScale
        return matches

    def presenceVector(self, text):
        pos = np.zeros(self.numTags, dtype=np.float32)
        neg = np.zeros(self.numTags, dtype=np.float32)
        negate = False
        for token in self.nlp(text.lower()):
            if token.is_punct or token.is_space:
                negate = False
                continue
            if token.lower_ in NEGATIONS or token.lower_ == "n't":
                negate = True
                continue
            if token.is_stop:
                continue
            hits = self._termMatches(token)
            target = neg if negate else pos
            for i, magnitude in hits.items():
                target[i] = max(target[i], magnitude)
            negate = False
        return pos - neg

    def presenceVectors(self, texts, batchSize=256):
        """Batched version of presenceVector using nlp.pipe for throughput."""
        results = []
        docs = self.nlp.pipe([t.lower() for t in texts], batch_size=batchSize)
        for doc in docs:
            pos = np.zeros(self.numTags, dtype=np.float32)
            neg = np.zeros(self.numTags, dtype=np.float32)
            negate = False
            for token in doc:
                if token.is_punct or token.is_space:
                    negate = False
                    continue
                if token.lower_ in NEGATIONS or token.lower_ == "n't":
                    negate = True
                    continue
                if token.is_stop:
                    continue
                hits = self._termMatches(token)
                target = neg if negate else pos
                for i, magnitude in hits.items():
                    target[i] = max(target[i], magnitude)
                negate = False
            results.append(pos - neg)
        return results
