"""Suggested tags for query words the search does not know, from WordNet and Datamuse.

For every adjective and noun that the search does not already understand, this collects related words, checks each
against what the search itself does with a single word (the reviewed alias table, Porter2 stems, the caption table) and
keeps the tags they reach. Two sources of related words:

    WordNet   synonyms (words in the same sense), "similar to" (adjectives), "see also". A word's senses are weighted by how
              often WordNet's sense-tagged text uses each, so a rare sense ("wet" as in drunk) counts for almost nothing.
              Verbs and adverbs are skipped: "fancy" the verb is "imagine", not a style.
    Datamuse  (https://www.datamuse.com/api/, offline, through a local cache) words that mean like the query word, and for a
              noun the adjectives that are often used to modify it ("cheese" -> soft, sharp, old; Google Books n-grams).
              Datamuse knows nothing about senses, so a means-like word that WordNet ties only to a rare sense of the query
              word keeps a share of its weight (SOFT_FLOOR), not all of it.

A related word's weight is its source weight (WordNet: relation x sense share; Datamuse: a gently falling function of its
rank), and tags combine as a noisy-or over the related words that reach them:

    score(tag) = 1 - prod over related words (1 - weight * tagWeight)

Antonyms are the main way a thesaurus goes wrong (clean -> sloppy), so related words are dropped when they sit on the
opposite side of the query word: Datamuse's antonyms of it, the words that mean like those antonyms, the words whose own
means-like list contains one of them, and WordNet's antonym cluster (the opposite head adjective and its satellites).

The result is a small table, configs/synonymTags.json, that the server reads; nothing here runs on the server.

    pip install nltk wordfreq && python -m nltk.downloader wordnet
    python site/tools/buildSynonyms.py                          # WordNet + Datamuse (needs network the first time)
    python site/tools/buildSynonyms.py --offline                # same, from build/datamuse.sqlite only
    python site/tools/buildSynonyms.py --noDatamuse             # WordNet only, as before
    python site/tools/buildSynonyms.py --examples wet old fancy --noWrite   # just those words (a few dozen requests)

Datamuse is asked about the words in WordNet that people actually use (wordfreq zipf >= --minZipf, default 3.0) and that
the search does not already understand, a few requests each; the answers are cached, so a build that stops carries on.
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

# Datamuse. A means-like word at rank i counts 1 / (1 + i / ML_FLATNESS): the first thirty are all close in meaning, so the
# weight barely falls. A modifier of a noun at rank i counts JJB_WEIGHT / (1 + i / JJB_DECAY): those are ordered by how
# often they are used, which says less about how well they fit.
ML_N, ML_FLATNESS = 30, 50
JJB_N, JJB_WEIGHT, JJB_DECAY = 40, 0.8, 10
SOFT_FLOOR = 0.3            # a means-like word WordNet ties only to a rare sense of the query word keeps at least this share
ANT_N, ANT_CHECKED = 12, 4  # antonyms asked for; the first few define "the opposite side"
ML_NEIGHBOURS, SYN_N = 15, 10


class Unavailable(Exception):
    """Datamuse data this word needs is neither cached nor fetchable (offline, or the request failed)."""


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
    p.add_argument("--minSenseWeight", type=float, default=0.05, help="ignore WordNet senses with less of the word's use")
    p.add_argument("--noDatamuse", action="store_true", help="WordNet only")
    p.add_argument("--cache", default=os.path.join("build", "datamuse.sqlite"), help="where Datamuse's answers are kept")
    p.add_argument("--offline", action="store_true", help="answer from the Datamuse cache only; words it lacks are skipped")
    p.add_argument("--workers", type=int, default=6, help="parallel Datamuse requests")
    p.add_argument("--minZipf", type=float, default=3.0,
                   help="with Datamuse, only words at least this common (wordfreq zipf scale; 3 is one per million words, "
                        "0 turns the filter off)")
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
    def __init__(self, args, wordnet, datamuse=None):
        """datamuse: a Datamuse client (site/tools/datamuse.py) to add its words to WordNet's; None for WordNet only."""
        self.args, self.wn, self.datamuse = args, wordnet, datamuse
        module = loadVocabularyModule()
        modelTags = loadModelTags(args.modelVocab)
        self.vocabulary = module.TagVocabulary(args.vocabulary, extraTags=modelTags)
        modelSet = set(modelTags)
        # groups the model can score, as the server computes them
        self.groups = {name for name, entry in self.vocabulary.canonical.items()
                       if any(m in modelSet for m in entry["members"])}
        self.tagsOf = lru_cache(maxsize=None)(self._tagsOf)
        self.wordnetView = lru_cache(maxsize=None)(self._wordnetView)

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

    def understood(self, word):
        """Does the search already do something with this word (so it needs no suggestion)?"""
        return bool(self.tagsOf(word)) or word in self.vocabulary.aliases or word in self.groups

    # ---- WordNet

    def senseWeights(self, word):
        """{synset: share of the word's use}, add-one smoothed so a sense WordNet's text never uses still counts a bit"""
        senses = [s for s in self.wn.synsets(word) if s.pos() in KEEP_POS]
        counts = [sum(l.count() for l in s.lemmas() if l.name().lower() == word) for s in senses]
        total = sum(counts) + len(senses)
        return {s: (c + 1) / total for s, c in zip(senses, counts)}

    def related(self, sense):
        """[(relation, synset)] for a sense: itself, "similar to" and "see also" (a few pointers in the data dangle)"""
        group = [("syn", sense)] + [("similar", x) for x in sense.similar_tos()] + [("also", x) for x in sense.also_sees()]
        return [(relation, other) for relation, other in group if other is not None]

    def suggestWordNet(self, word):
        """[(tag, score, best related word)], best first. Each sense gets one vote per tag, from its best related word, and
        senses combine as a noisy-or, so three words from one wrong sense do not outweigh one word from the right one."""
        miss, via = {}, {}
        for sense, weight in self.senseWeights(word).items():
            if weight < self.args.minSenseWeight:
                continue
            best = {}
            for relation, other in self.related(sense):
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

    def _wordnetView(self, word):
        """What WordNet says about a word, as (direct, attach, opposite):
        direct    {related word: weight} -- relation weight x sense share, for the senses that matter;
        attach    {related word: share of the best sense it hangs off} -- over every sense, however rare;
        opposite  the antonym cluster: the antonyms of the word's senses and the satellites around them."""
        direct, attach, opposite = {}, {}, set()
        for sense, weight in self.senseWeights(word).items():
            for relation, other in self.related(sense):
                for lemma in other.lemmas():
                    name = lemma.name().lower()
                    if name == word or not WORD.fullmatch(name):
                        continue
                    attach[name] = max(attach.get(name, 0.0), weight)
                    if weight >= self.args.minSenseWeight:
                        direct[name] = max(direct.get(name, 0.0), RELATION[relation] * weight)
            for lemma in sense.lemmas():
                for antonym in lemma.antonyms():
                    for cluster in [antonym.synset()] + antonym.synset().similar_tos():
                        if cluster is not None:
                            opposite.update(l.name().lower() for l in cluster.lemmas())
        return direct, attach, opposite

    # ---- Datamuse

    def ask(self, rel, word, n=ML_N):
        found = self.datamuse.get(rel, word, n)
        if found is None:
            raise Unavailable(f"{rel}({word})")
        return found

    def partsOfSpeech(self, word):
        return set(self.ask("pos", word, 1))

    def gather(self, word):
        """{related word: (weight, source)} from WordNet and Datamuse, before antonyms are dropped."""
        direct, attach, _ = self.wordnetView(word)
        found = {}

        def add(candidate, weight, source):
            if candidate != word and WORD.fullmatch(candidate) and weight > found.get(candidate, (0, ""))[0]:
                found[candidate] = (weight, source)

        for candidate, weight in direct.items():
            add(candidate, weight, "wordnet")
        parts = self.partsOfSpeech(word)
        if "adj" in parts or not parts:
            for i, candidate in enumerate(self.ask("ml", word, ML_N)):
                gate = max(attach.get(candidate, 1.0), SOFT_FLOOR)
                add(candidate, gate / (1 + i / ML_FLATNESS), "ml")
        if "n" in parts and "adj" not in parts:
            for i, candidate in enumerate(self.ask("rel_jjb", word, JJB_N)):
                add(candidate, JJB_WEIGHT / (1 + i / JJB_DECAY), "jjb")
        return found

    def antonymsOf(self, word):
        return self.ask("rel_ant", word, ANT_N)[:ANT_CHECKED]

    def oppositeSide(self, word):
        """Words on the other side of the query word: its antonyms and what means like them, plus WordNet's cluster."""
        side = set(self.wordnetView(word)[2])
        for antonym in self.antonymsOf(word):
            side.add(antonym)
            side.update(self.ask("ml", antonym, ML_N)[:ML_NEIGHBOURS])
            side.update(self.ask("rel_syn", antonym, SYN_N))
        return side

    def candidates(self, word):
        """({related word: (weight, source)}, {dropped related word: reason}) -- raises Unavailable if Datamuse data is missing.
        The "means like an antonym" check is only made for words that reach a tag: it costs a request each, and a word
        that reaches no tag cannot change the result."""
        found = self.gather(word)
        antonyms = self.antonymsOf(word)
        side = self.oppositeSide(word)
        dropped = {}
        for candidate in list(found):
            if candidate in side:
                dropped[candidate] = "opposite side"
            elif antonyms and self.tagsOf(candidate) and any(a in self.ask("ml", candidate, ML_N)[:ML_NEIGHBOURS] for a in antonyms):
                dropped[candidate] = "means like an antonym"
        return {c: v for c, v in found.items() if c not in dropped}, dropped

    def score(self, related):
        miss, via = {}, {}
        for candidate, (weight, _) in related.items():
            for tag, tagWeight in self.tagsOf(candidate).items():
                value = weight * tagWeight
                miss[tag] = miss.get(tag, 1.0) * (1 - value)
                if value >= via.get(tag, (0, ""))[0]:
                    via[tag] = (value, candidate)
        ranked = sorted(((tag, 1 - m) for tag, m in miss.items()), key=lambda kv: (-kv[1], kv[0]))
        return [(tag, score, via[tag][1]) for tag, score in ranked if score >= self.args.minScore][:self.args.topK]

    def suggest(self, word):
        """[(tag, score, best related word)], best first; None when Datamuse data for the word is missing."""
        if self.datamuse is None:
            return self.suggestWordNet(word)
        try:
            related, _ = self.candidates(word)
        except Unavailable:
            return None
        return self.score(related)

    def prefetch(self, words):
        """Ask Datamuse everything suggest() will need for these words, in three rounds (each depends on the last)."""
        if self.datamuse is None:
            return
        words = list(dict.fromkeys(words))
        self.datamuse.prefetch([("pos", w, 1) for w in words], "part of speech")
        second = []
        for w in words:
            try:
                parts = self.partsOfSpeech(w)
            except Unavailable:
                continue
            second.append(("rel_ant", w, ANT_N))
            if "adj" in parts or not parts:
                second.append(("ml", w, ML_N))
            if "n" in parts and "adj" not in parts:
                second.append(("rel_jjb", w, JJB_N))
        self.datamuse.prefetch(second, "related words")
        third = []
        for w in words:
            try:
                found = self.gather(w)
                antonyms = self.antonymsOf(w)
            except Unavailable:
                continue
            for a in antonyms:
                third += [("ml", a, ML_N), ("rel_syn", a, SYN_N)]
            if antonyms:
                third += [("ml", c, ML_N) for c in found if self.tagsOf(c)]   # a word that reaches no tag cannot change the result
        self.datamuse.prefetch(third, "antonym checks")

    def words(self):
        names = set()
        for pos in ("a", "s", "n"):
            names.update(n for n in self.wn.all_lemma_names(pos) if WORD.fullmatch(n) and len(n) > 2)
        return sorted(names)


def commonEnough(names, minZipf):
    """The words at least minZipf on wordfreq's scale (3.0 is one per million words)."""
    if minZipf <= 0:
        return names
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        sys.exit("needs wordfreq for --minZipf: pip install wordfreq (or pass --minZipf 0 to try every WordNet word)")
    return [n for n in names if zipf_frequency(n, "en") >= minZipf]


def main():
    args = parseArgs()
    datamuse = None
    if not args.noDatamuse:
        from datamuse import Datamuse
        datamuse = Datamuse(args.cache, offline=args.offline, workers=args.workers)
    builder = Builder(args, loadWordNet(), datamuse)
    try:
        # the search already understands these; suggestions are only for words that match nothing
        names = [w for w in builder.words() if not builder.understood(w)]
        if args.examples and args.noWrite:
            names = []                       # just trying a few words: do not build the table around them
        elif datamuse is not None:
            names = commonEnough(names, args.minZipf)
            print(f"{len(names)} words to look up", file=sys.stderr)
        if datamuse is not None:
            builder.prefetch(names + [w for w in args.examples if w not in names])
        table, skipped = {}, 0
        for i, word in enumerate(names):
            if i % 5000 == 0:
                print(f"{i}/{len(names)}", file=sys.stderr)
            found = builder.suggest(word)
            if found is None:
                skipped += 1
            elif found:
                table[word] = [[tag, round(score, 3), via] for tag, score, via in found]
        print(f"{len(table)} words with suggestions" + (f", {skipped} skipped for lack of Datamuse data" if skipped else ""),
              file=sys.stderr)

        for word in args.examples:
            found = builder.suggest(word)
            print(f"{word}: " + ("(no Datamuse data)" if found is None else
                                 ", ".join(f"{t} {s:.2f} (via {v})" for t, s, v in found) or "-"))

        if not args.noWrite:
            source = "WordNet" if datamuse is None else "WordNet + Datamuse"
            settings = {"minScore": args.minScore, "topK": args.topK, "minSenseWeight": args.minSenseWeight,
                        "relationWeights": RELATION, "pos": KEEP_POS}
            if datamuse is not None:
                settings["datamuse"] = {"minZipf": args.minZipf, "mlFlatness": ML_FLATNESS, "jjbWeight": JJB_WEIGHT,
                                        "jjbDecay": JJB_DECAY, "softFloor": SOFT_FLOOR, "antonymsChecked": ANT_CHECKED}
            out = {"source": f"{source} (site/tools/buildSynonyms.py)", "settings": settings, "words": table}
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)", file=sys.stderr)
    finally:
        if datamuse is not None:
            datamuse.close()


if __name__ == "__main__":
    main()
