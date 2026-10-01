"""Word -> tag associations learned from the LLM caption sets, for query words the reviewed vocabulary does not know.

The captions in results/fontQueries.json and results/fontQueriesV2.json were generated from each MyFonts font's real
tags, so a word the captions keep using for fonts with a given tag ("wet" for fonts tagged drip / liquid) says what
users are likely to mean by it. For every caption word and every searchable tag group (a canonical tag of
configs/tagVocabulary.json, or a model tag searchable by its own name) this counts, over MyFonts fonts, how many fonts
carry both, and keeps the groups the word is most over-represented with:

    lift = P(group | word) / P(group), shrunk toward 1:  (n_wg + a * P(g)) / ((n_w + a) * P(g))

Only fonts with a MyFonts tag file are used (DaFont captions were written from two words and are excluded).

    python site/tools/buildWordTags.py                  # writes configs/wordTags.json and prints the evaluation
    python site/tools/buildWordTags.py --evaluate-only  # held-out check against the reviewed aliases, no file

Evaluation: every single-word alias of the reviewed vocabulary (weight >= 0.8) whose word is not itself a tag name is
held-out ground truth: does the table, built from captions alone, put the alias's canonical tag first / in its top 3?
"""
import argparse
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def loadVocabularyModule():
    spec = importlib.util.spec_from_file_location("tagVocabulary", os.path.join(REPO, "utils", "tagVocabulary.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", nargs="+", default=[os.path.join("results", "fontQueries.json"),
                                                      os.path.join("results", "fontQueriesV2.json")])
    p.add_argument("--tagDir", default=os.path.join("dataset", "taglabel"))
    p.add_argument("--vocabulary", default=os.path.join("configs", "tagVocabulary.json"))
    p.add_argument("--modelVocab", default=os.path.join("site", "backend", "data", "vocab.json"),
                   help="the tagger's tag list (bundle vocab.json); groups the model cannot score are left out")
    p.add_argument("--out", default=os.path.join("configs", "wordTags.json"))
    p.add_argument("--minFonts", type=int, default=5, help="a word must appear in captions of this many fonts")
    p.add_argument("--minJoint", type=int, default=3, help="fonts with both the word and the group")
    p.add_argument("--prior", type=float, default=10.0, help="shrinkage strength a")
    p.add_argument("--minLift", type=float, default=2.0)
    p.add_argument("--topK", type=int, default=3)
    p.add_argument("--score", choices=["lift", "logodds"], default="logodds",
                   help="rank groups by shrunk lift, or by the log-odds z-score with an informative Dirichlet prior "
                        "(Monroe et al. 2008), which does not let a tag seen on a handful of fonts win on lift alone")
    p.add_argument("--minZ", type=float, default=3.0, help="logodds: minimum z-score")
    p.add_argument("--alpha", type=float, default=100.0, help="logodds: prior strength")
    p.add_argument("--minGroupFonts", type=int, default=30, help="ignore tag groups on fewer MyFonts fonts")
    p.add_argument("--exclude", default=os.path.join("configs", "wordTagsExclude.txt"),
                   help="reviewed list of words that are not style words (one per line, # comments)")
    p.add_argument("--evaluate-only", dest="evaluateOnly", action="store_true")
    return p.parse_args()


def main():
    args = parseArgs()
    os.chdir(REPO)
    tv = loadVocabularyModule()
    with open(args.modelVocab, encoding="utf-8") as f:
        modelTags = json.load(f)
    vocabulary = tv.TagVocabulary(args.vocabulary, extraTags=modelTags)
    modelSet = set(modelTags)
    groupOf = {}
    for name, entry in vocabulary.canonical.items():
        members = [m for m in entry["members"] if m in modelSet]
        for m in members:
            groupOf.setdefault(m, name)

    # font -> groups (from its MyFonts tags) and font -> caption words
    wordsOf = defaultdict(set)
    for path in args.captions:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for font, captions in data.items():
            for caption in (captions if isinstance(captions, list) else [captions]):
                wordsOf[font].update(t for t in tv.normalize(str(caption)) if re.fullmatch(r"[a-z]{3,}", t))
    groupsOf = {}
    for font in wordsOf:
        path = os.path.join(args.tagDir, font)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                groups = {groupOf[t] for t in f.read().split() if t in groupOf}
            if groups:
                groupsOf[font] = groups
    fonts = list(groupsOf)
    N = len(fonts)
    groupCount = Counter(g for font in fonts for g in groupsOf[font])
    from spacy.lang.en.stop_words import STOP_WORDS    # general English function words ("using", "with", ...)
    excluded = set(tv.STOPWORDS) | set(STOP_WORDS)
    if args.exclude and os.path.exists(args.exclude):
        with open(args.exclude, encoding="utf-8") as f:
            excluded |= {line.split("#")[0].strip() for line in f if line.split("#")[0].strip()}
    wordCount = Counter(w for font in fonts for w in wordsOf[font] if w not in excluded)
    joint = defaultdict(Counter)
    for font in fonts:
        groups = groupsOf[font]
        for w in wordsOf[font]:
            if wordCount.get(w, 0) >= args.minFonts:
                joint[w].update(groups)
    print(f"{N} MyFonts fonts with captions and tags, {len(groupCount)} tag groups, "
          f"{len(joint)} words in captions of >= {args.minFonts} fonts", flush=True)

    table = buildTable(joint, wordCount, groupCount, N, args)

    result = evaluate(vocabulary, modelSet, table, args)
    for w in ("wet", "liquid", "melting", "slimy", "gooey", "dripping", "diner", "greasy", "gross", "spooky",
              "whimsical", "sleek", "cozy", "rugged", "techy"):
        print(f"  {w:10s} -> {table.get(w, '(not in table)')}")

    if not args.evaluateOnly:
        header = {"source": "caption co-occurrence (site/tools/buildWordTags.py)", "fonts": N,
                  "settings": {k: getattr(args, k) for k in ("score", "minFonts", "minJoint", "minZ", "alpha",
                                                            "minGroupFonts", "minLift", "prior", "topK")}}
        # one word per line: [[group, score, fonts with both], ...], score = z (logodds) or log lift
        lines = [f"{json.dumps(w)}: {json.dumps(table[w])}" for w in sorted(table)]
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(header)[:-1] + ', "words": {\n' + ",\n".join(lines) + "\n}}\n")
        print(f"wrote {len(table)} words to {args.out}")


def buildTable(joint, wordCount, groupCount, N, args):
    table = {}
    for w, counts in joint.items():
        nw = wordCount[w]
        scored = []
        for g, nwg in counts.items():
            ng = groupCount[g]
            if nwg < args.minJoint or ng < args.minGroupFonts:
                continue
            pg = ng / N
            lift = (nwg + args.prior * pg) / ((nw + args.prior) * pg)
            if lift < args.minLift:
                continue
            if args.score == "lift":
                scored.append((g, math.log(lift), nwg))
            else:
                a = args.alpha * pg                      # prior pseudo-count for this group
                inside = math.log((nwg + a) / (nw - nwg + args.alpha - a))
                outside = math.log((ng + a) / (N - ng + args.alpha - a))
                z = (inside - outside) / math.sqrt(1 / (nwg + a) + 1 / (ng + a))
                if z >= args.minZ:
                    scored.append((g, z, nwg))
        scored.sort(key=lambda x: -x[1])
        if scored:
            table[w] = [[g, round(v, 3), n] for g, v, n in scored[:args.topK]]
    return table


def evaluate(vocabulary, modelSet, table, args):
    """Held-out check on the reviewed aliases: single words, weight >= 0.8, not themselves a tag or canonical name."""
    names = set(vocabulary.canonical) | modelSet
    stems = vocabulary.aliasStems
    cases = []
    for canon, entry in vocabulary.canonical.items():
        if entry.get("facet") == "other":
            continue
        for phrase, weight in entry["aliases"].items():
            # words the parser already reaches through another phrase's stem (serifs -> serif) are not the table's job
            sameStem = stems.get(vocabulary.stem(phrase), []) if vocabulary.stem else []
            if weight >= 0.8 and re.fullmatch(r"[a-z]+", phrase) and phrase not in names \
                    and not any(w != phrase for w in sameStem):
                cases.append((phrase, canon))
    covered = [(w, c) for w, c in cases if w in table]
    top1 = sum(table[w][0][0] == c for w, c in covered)
    top3 = sum(c in [g for g, _, _ in table[w]] for w, c in covered)
    print(f"held-out aliases: {len(cases)}; in table {len(covered)} ({len(covered) / max(1, len(cases)):.0%}); "
          f"of those, correct first {top1} ({top1 / max(1, len(covered)):.0%}), in top {args.topK} {top3} "
          f"({top3 / max(1, len(covered)):.0%})")
    misses = [(w, c, [g for g, _, _ in table[w]]) for w, c in covered if c not in [g for g, _, _ in table[w]]]
    print("  sample misses:", misses[:12])
    return {"cases": len(cases), "covered": len(covered), "top1": top1, "top3": top3}


if __name__ == "__main__":
    main()
