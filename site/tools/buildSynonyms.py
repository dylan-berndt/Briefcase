"""Suggested tags for query words the search does not know, from WordNet.

WordNet is a hand-built thesaurus (Princeton): no web text, so none of the associations a word-vector model picks up
from it. For every adjective and noun in it that the search does not already understand, this looks at the words
WordNet relates to it, checks each against what the search itself does with a single word (the reviewed alias table,
Porter2 stems, the caption table), and keeps the tags they reach:

    synonyms (words in the same sense), "similar to" (adjectives), "see also"

Only the first senses of a word count, in proportion to how often WordNet's sense-tagged text uses each (verbs and
adverbs are skipped: "fancy" the verb is "imagine", not a style). Each sense gets one vote per tag, from its best
related word, and the votes of different senses are combined as a noisy-or, so three words from one wrong sense do not
outweigh one word from the right one:

    score(tag) = 1 - prod over senses (1 - senseWeight * relationWeight * tagWeight)

The result is a small table, configs/synonymTags.json, that the server reads; nothing here runs on the server.

    pip install nltk && python -m nltk.downloader wordnet
    python site/tools/buildSynonyms.py                         # writes configs/synonymTags.json
    python site/tools/buildSynonyms.py --examples wet old fancy  # also prints those words' suggestions

The tagger's tag list (site/backend/data/vocab.json, a git-lfs file, so it has to be pulled) decides which tags can be
suggested; a tag the model cannot score is left out. Rebuild when the vocabulary, the caption table or the model's tags
change.
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

RELATION = {"syn": 1.0, "similar": 0.7, "also": 0.5}
KEEP_POS = "asn"            # adjectives (a, satellite s) and nouns
INFERRED_WEIGHT = 0.5       # the caption table's guesses count as the search counts them
WORD = re.compile(r"[a-z]+")


def loadVocabularyModule():
    spec = importlib.util.spec_from_file_location("tagVocabulary", os.path.join(REPO, "utils", "tagVocabulary.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--vocabulary", default=os.path.join("configs", "tagVocabulary.json"))
    p.add_argument("--modelVocab", default=os.path.join("site", "backend", "data", "vocab.json"),
                   help="the tagger's tag list (bundle vocab.json); tags the model cannot score are left out")
    p.add_argument("--out", default=os.path.join("configs", "synonymTags.json"))
    p.add_argument("--minScore", type=float, default=0.05, help="drop suggestions scoring below this")
    p.add_argument("--topK", type=int, default=8, help="suggestions kept per word")
    p.add_argument("--minSenseWeight", type=float, default=0.05, help="ignore senses with less of the word's use")
    p.add_argument("--examples", nargs="*", default=[], help="print these words' suggestions")
    p.add_argument("--noWrite", action="store_true")
    return p.parse_args()


def loadWordNet():
    try:
        import nltk
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
    except ImportError:
        sys.exit("needs nltk: pip install nltk")
    except LookupError:
        sys.exit("needs the WordNet data: python -m nltk.downloader wordnet")
    return wordnet


def loadModelTags(path):
    with open(path, encoding="utf-8") as f:
        head = f.read(200)
        f.seek(0)
        if head.startswith("version https://git-lfs"):
            sys.exit(f"{path} is a git-lfs pointer; pull the real file (git lfs pull) or pass --modelVocab")
        return json.load(f)


class Builder:
    def __init__(self, args, wordnet):
        self.args, self.wn = args, wordnet
        module = loadVocabularyModule()
        modelTags = loadModelTags(args.modelVocab)
        self.vocabulary = module.TagVocabulary(args.vocabulary, extraTags=modelTags)
        modelSet = set(modelTags)
        # groups the model can score, as the server computes them
        self.groups = {name for name, entry in self.vocabulary.canonical.items()
                       if any(m in modelSet for m in entry["members"])}
        self.tagsOf = lru_cache(maxsize=None)(self._tagsOf)

    def _tagsOf(self, word):
        """What the search does with this single word today: {tag group: weight}."""
        found = {}
        weights, _ = self.vocabulary.parse(word, inferred=found)
        out = {g: w for g, w in weights.items() if w > 0 and g in self.groups}
        for _, (sign, groups) in found.items():
            if sign < 0:
                continue
            for g in groups[:2]:
                if g in self.groups:
                    out.setdefault(g, INFERRED_WEIGHT)
        return out

    def senseWeights(self, word):
        """{synset: share of the word's use}, add-one smoothed so a sense WordNet's text never uses still counts a bit"""
        senses = [s for s in self.wn.synsets(word) if s.pos() in KEEP_POS]
        counts = [sum(l.count() for l in s.lemmas() if l.name().lower() == word) for s in senses]
        total = sum(counts) + len(senses)
        return {s: (c + 1) / total for s, c in zip(senses, counts)}

    def suggest(self, word):
        """[(tag, score, best related word)], best first"""
        miss, via = {}, {}
        for sense, weight in self.senseWeights(word).items():
            if weight < self.args.minSenseWeight:
                continue
            group = [("syn", sense)] + [("similar", x) for x in sense.similar_tos()] \
                + [("also", x) for x in sense.also_sees()]
            best = {}
            for relation, other in group:
                for lemma in other.lemmas():
                    name = lemma.name().lower()
                    if name == word or not WORD.fullmatch(name):
                        continue
                    for tag, tagWeight in self.tagsOf(name).items():
                        value = RELATION[relation] * tagWeight
                        if value > best.get(tag, (0, ""))[0]:
                            best[tag] = (value, name)
            for tag, (value, name) in best.items():
                miss[tag] = miss.get(tag, 1.0) * (1 - weight * value)
                if value * weight >= via.get(tag, (0, ""))[0]:
                    via[tag] = (value * weight, name)
        ranked = sorted(((tag, 1 - m) for tag, m in miss.items()), key=lambda kv: (-kv[1], kv[0]))
        return [(tag, score, via[tag][1]) for tag, score in ranked if score >= self.args.minScore][:self.args.topK]

    def words(self):
        names = set()
        for pos in ("a", "s", "n"):
            names.update(n for n in self.wn.all_lemma_names(pos) if WORD.fullmatch(n) and len(n) > 2)
        return sorted(names)


def main():
    args = parseArgs()
    builder = Builder(args, loadWordNet())
    table = {}
    names = builder.words()
    for i, word in enumerate(names):
        if i % 20000 == 0:
            print(f"{i}/{len(names)}", file=sys.stderr)
        if builder.tagsOf(word) or word in builder.vocabulary.aliases or word in builder.groups:
            continue          # the search already understands it; suggestions are only for words that match nothing
        found = builder.suggest(word)
        if found:
            table[word] = [[tag, round(score, 3), via] for tag, score, via in found]
    print(f"{len(table)} words with suggestions", file=sys.stderr)

    for word in args.examples:
        print(f"{word}: " + (", ".join(f"{t} {s:.2f} (via {v})" for t, s, v in builder.suggest(word)) or "-"))

    if not args.noWrite:
        out = {"source": "WordNet (site/tools/buildSynonyms.py)",
               "settings": {"minScore": args.minScore, "topK": args.topK, "minSenseWeight": args.minSenseWeight,
                            "relationWeights": RELATION, "pos": KEEP_POS},
               "words": table}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
