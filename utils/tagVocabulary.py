# Deterministic free-text query -> canonical tag weights, driven by configs/tagVocabulary.json
# (built by experiments/critical-review/build_tag_vocabulary.py). No embeddings, no synonyms
# guessed at query time: every phrase that maps to a tag is listed in the reviewed alias table.
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


class TagVocabulary:
    def __init__(self, path=os.path.join("configs", "tagVocabulary.json")):
        with open(path) as f:
            data = json.load(f)
        self.canonical = data["canonical"]
        self.dropped = data.get("dropped", {})

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

        self.children = {}
        for canon, entry in self.canonical.items():
            for parent in entry["implies"]:
                self.children.setdefault(parent, []).append(canon)

    def parse(self, query, expandChildren=0.0):
        """Returns ({canonical: weight}, unmatchedWords). Negated tags get negative weight.
        expandChildren > 0 also adds tags that imply a queried tag (e.g. 'serif' -> 'didone')
        at that fraction of the parent's weight."""
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
                if tok not in STOPWORDS and not re.fullmatch(r"[,;.!?]", tok):
                    unmatched.append(tok)
                i += 1

        if expandChildren > 0:
            for parent, w in list(weights.items()):
                if w <= 0:
                    continue
                for child in self.children.get(parent, []):
                    weights.setdefault(child, w * expandChildren)
        return weights, unmatched
