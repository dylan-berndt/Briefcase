"""Suggested tags for query words that match nothing, from a WordNet + Datamuse table.

configs/synonymTags.json (built offline by site/tools/buildSynonyms.py) maps a word to the tags that the words WordNet
and Datamuse relate to it (synonyms, means-like words, adjectives that modify a noun; antonyms are dropped) lead to,
each with a score and the related word it came through.
Nothing is computed here: an unknown query word is looked up, as typed or by its Porter2 stem ("drenched" finds
"drench"), and the tags the model can score are offered. Nothing here affects the ranking: the user decides whether to
add a suggested tag.
"""

import json
import logging
import os

log = logging.getLogger(__name__)


class SynonymTable:
    def __init__(self, path, groups, stem=None, topK=8, minScore=0.05):
        """groups: the tag groups the index can score. stem: the parser's word -> Porter2 stem, or None."""
        with open(path, encoding="utf-8") as f:
            table = json.load(f)["words"]
        self.table = {}
        for word, entries in table.items():
            usable = [(tag, score, via) for tag, score, via in entries if tag in groups and score >= minScore]
            if usable:
                self.table[word] = usable
        self.topK, self.stem = topK, stem
        self.byStem = {}
        if stem is not None:
            for word in sorted(self.table):
                if word.isalpha():
                    self.byStem.setdefault(stem(word), word)

    def suggest(self, word, exclude=()):
        """[(tag, via word, score)], best first; [] for a word the table does not have."""
        entries = self.table.get(word)
        if entries is None and self.stem is not None:
            match = self.byStem.get(self.stem(word))
            entries = self.table.get(match) if match else None
        skip = set(exclude)
        return [(tag, via, score) for tag, score, via in entries or () if tag not in skip][:self.topK]


def findSynonymTable():
    here = os.path.dirname(os.path.abspath(__file__))
    for directory in (os.path.join(here, "configs"), os.path.join(os.path.dirname(os.path.dirname(here)), "configs")):
        path = os.path.join(directory, "synonymTags.json")
        if os.path.exists(path):
            return path
    return None


def loadSuggester(groups, path="auto", stem=None):
    """The suggester, or None when it is switched off (path "") or the table is missing (logged)."""
    if path == "":
        return None
    path = findSynonymTable() if path == "auto" else path
    if not path or not os.path.exists(path):
        log.warning("synonym suggestions are off: configs/synonymTags.json not found")
        return None
    return SynonymTable(path, groups, stem)
