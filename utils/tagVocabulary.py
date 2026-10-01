# Deterministic free-text query -> canonical tag weights, driven by configs/tagVocabulary.json
# (built by experiments/critical-review/build_tag_vocabulary.py). No embeddings, no synonyms
# guessed at query time: every phrase that maps to a tag is listed in the reviewed alias table, and an
# unmatched word only matches a single-word phrase with the same Porter2 stem (dripping -> drip, swirling -> swirls).
# Not imported by utils/__init__ on purpose (no torch dependency needed here).

import json
import os
import re

NEGATORS = {"not", "no", "without", "non", "never", "nothing", "avoid", "except"}
# A negation covers the next matched phrase only, extended through "or"/"nor" lists
# ("not bold thin" -> -bold +thin; "no serifs or swashes" -> -serif -swash)
NEGATION_CONTINUERS = {"or", "nor"}
STOPWORDS = {"a", "an", "the", "font", "fonts", "typeface", "typefaces", "type", "style", "styled",
             "looking", "look", "looks", "like", "that", "is", "are", "for", "of", "in", "on", "to",
             "something", "very", "kind", "sort", "some", "bit", "little", "really", "quite", "i", "want",
             "need", "please", "inspired", "based", "esque", "themed", "theme", "letters", "lettering",
             "text", "feel", "feeling", "vibe", "ish", "y", "and", "or", "nor", "but", "with", "yet"}


def normalize(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[-_/'&]", " ", text)
    text = re.sub(r"([,;.!?])", r" \1 ", text)
    return text.split()


def loadStemmer():
    """The Porter2 (Snowball English) stemmer, or None when snowballstemmer is not installed (then words only match
    as typed)."""
    try:
        import snowballstemmer
    except ImportError:
        return None
    return snowballstemmer.stemmer("english").stemWord


def stemIndex(words, stem):
    """{stem: [words]} over single words, for matching an unknown word to known words that share its stem."""
    index = {}
    if stem is not None:
        for word in sorted(words):
            if word.isalpha():
                index.setdefault(stem(word), []).append(word)
    return index


def closest(word, candidates):
    """Of several known words sharing a stem, the one sharing the longest prefix with the word, then the shortest
    ("classically" -> classical rather than classic)."""
    prefix = lambda c: len(os.path.commonprefix([c, word]))   # noqa: E731
    return max(candidates, key=lambda c: (prefix(c), -len(c)))


# Model tags that are not in the reviewed vocabulary can still be searched by their own name, unless it is junk
# (URL-encoded non-Latin entries show up in the MyFonts vocab)
PLAIN_TAG = re.compile(r"[a-z0-9][a-z0-9 \-]*")


class TagVocabulary:
    def __init__(self, path=os.path.join("configs", "tagVocabulary.json"), extraTags=None, wordTags="auto",
                 stemmer="auto"):
        """extraTags: every tag the tagger predicts. Those the reviewed vocabulary neither merges into a canonical
        nor drops become single-member canonicals named after themselves, so all of the model's tags are searchable.
        wordTags: the caption co-occurrence table (site/tools/buildWordTags.py); "auto" loads wordTags.json next to
        the vocabulary file if there is one. It is only consulted by parse(..., inferred={}).
        stemmer: word -> stem, applied to both the query word and the known words; "auto" is Porter2 (the
        snowballstemmer package) if installed, None matches words only as typed."""
        self.stem = loadStemmer() if stemmer == "auto" else stemmer
        with open(path) as f:
            data = json.load(f)
        self.canonical = data["canonical"]
        self.dropped = data.get("dropped", {})

        if extraTags is not None:
            known = {m for entry in self.canonical.values() for m in entry["members"]} | set(self.dropped)
            for tag in extraTags:
                if tag not in known and tag not in self.canonical and PLAIN_TAG.fullmatch(tag):
                    self.canonical[tag] = {"facet": "other", "members": [tag], "aliases": {}, "implies": []}

        # phrase (tuple of normalized tokens) -> list of (canonical, weight)
        self.aliases = {}
        for canon, entry in self.canonical.items():
            phrases = dict(entry["aliases"])
            phrases.setdefault(canon, 1.0)
            for phrase, weight in phrases.items():
                key = tuple(normalize(phrase))
                targets = self.aliases.setdefault(key, [])
                existing = [k for k, (c, _) in enumerate(targets) if c == canon]
                if existing:
                    k = existing[0]
                    targets[k] = (canon, max(weight, targets[k][1]))
                else:
                    targets.append((canon, weight))
        self.maxPhrase = max(len(k) for k in self.aliases)
        self.aliasStems = stemIndex({k[0] for k in self.aliases if len(k) == 1}, self.stem)

        self.wordTags = {}
        if wordTags == "auto":
            wordTags = os.path.join(os.path.dirname(os.path.abspath(path)), "wordTags.json")
            wordTags = wordTags if os.path.exists(wordTags) else None
        if wordTags:
            with open(wordTags, encoding="utf-8") as f:
                table = json.load(f)["words"]
            for word, entries in table.items():
                groups = [entry[0] for entry in entries if entry[0] in self.canonical]
                if groups:
                    self.wordTags[word] = groups
        self.wordTagStems = stemIndex(self.wordTags, self.stem)

        self.children = {}
        for canon, entry in self.canonical.items():
            for parent in entry["implies"]:
                self.children.setdefault(parent, []).append(canon)

    def stemMatch(self, word, index):
        """The known word in a stemIndex sharing this word's stem, or None."""
        if self.stem is None or word in STOPWORDS:
            return None
        candidates = index.get(self.stem(word))
        return closest(word, candidates) if candidates else None

    def inferredGroups(self, word):
        """Tag groups the caption table associates with a word (or a word with its stem), best first; [] if none."""
        if word in self.wordTags:
            return self.wordTags[word]
        match = self.stemMatch(word, self.wordTagStems)
        return self.wordTags[match] if match else []

    def parse(self, query, expandChildren=0.0, inferred=None):
        """Returns ({canonical: weight}, unmatchedWords). Negated tags get negative weight.
        expandChildren > 0 also adds tags that imply a queried tag (e.g. 'serif' -> 'didone')
        at that fraction of the parent's weight.
        inferred: pass a dict to use the caption table for words nothing else matches; it is filled with
        {word: (sign, [groups])} and those words are left out of unmatchedWords. Left as None, behaviour is unchanged."""
        tokens = normalize(query)
        weights, unmatched = {}, []
        negated = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in NEGATORS:
                negated = True
                i += 1
                continue
            for n in range(min(self.maxPhrase, len(tokens) - i), 0, -1):
                key = tuple(tokens[i:i + n])
                if key in self.aliases:
                    for canon, w in self.aliases[key]:
                        w = -w if negated else w
                        # Keep the strongest evidence per tag; a negation always wins over a positive
                        if canon not in weights or abs(w) > abs(weights[canon]) or w < 0:
                            weights[canon] = w
                    i += n
                    if negated and not (i < len(tokens) and tokens[i] in NEGATION_CONTINUERS):
                        negated = False
                    break
            else:
                base = self.stemMatch(tok, self.aliasStems)
                if base is not None:
                    for canon, w in self.aliases[(base,)]:
                        w = -w if negated else w
                        if canon not in weights or abs(w) > abs(weights[canon]) or w < 0:
                            weights[canon] = w
                    if negated and not (i + 1 < len(tokens) and tokens[i + 1] in NEGATION_CONTINUERS):
                        negated = False
                elif inferred is not None and tok not in STOPWORDS and self.inferredGroups(tok):
                    inferred[tok] = (-1 if negated else 1, self.inferredGroups(tok))
                    if negated and not (i + 1 < len(tokens) and tokens[i + 1] in NEGATION_CONTINUERS):
                        negated = False
                elif tok not in STOPWORDS and not re.fullmatch(r"[,;.!?]", tok):
                    unmatched.append(tok)
                i += 1

        if expandChildren > 0:
            for parent, w in list(weights.items()):
                if w <= 0:
                    continue
                for child in self.children.get(parent, []):
                    weights.setdefault(child, w * expandChildren)
        return weights, unmatched
