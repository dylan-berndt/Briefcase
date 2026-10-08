"""One-off migration: regroup the tag vocabulary so a group is only ever a set of literal spelling variants of one tag.

Rules (everything else stays as the reviewed vocabulary has it: the dropped list, the facets):
  R1  Searchable tags are the ones searchable today: the reviewed vocabulary's members plus every other model tag that is not
      dropped. Two of them are one group only if they are equal once case, hyphens, spaces and punctuation are ignored
      (art-deco / artdeco, sci-fi / scifi). Nothing else merges: scary, spooky, dark, skull... are separate tags.
  R2  A group is named after the variant that was already a canonical name; failing that, the hyphenated variant; failing that,
      the first alphabetically.
  R3  A phrase from the old alias table that is itself a searchable tag (same letters ignoring case/punctuation) now points at
      that tag, at weight 1.0 ("dark" no longer means horror at 0.6; it means the tag dark).
  R4  Any other old alias phrase ("zombie", "gothic horror", "condensed") keeps its weight and points at the primary member of
      the group it used to belong to: the member that is the old canonical name, else the one with the most training fonts.
  R5  An old canonical name that is not a tag (condensed, rounded, wedding...) is itself added as an alias, under R3/R4.
  R6  `implies` follow the primary members.
The caption table (wordTags.json) is mapped the same way: an old group name becomes the new group of its primary member.
"""
import argparse
import collections
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser()
parser.add_argument("--vocabulary", default=os.path.join(REPO, "configs", "tagVocabulary.json"),
                    help="the reviewed vocabulary to regroup (the old, merged-groups one)")
parser.add_argument("--wordTags", default=os.path.join(REPO, "configs", "wordTags.json"), help="its caption table")
parser.add_argument("--modelVocab", default=os.path.join(REPO, "site", "backend", "data", "vocab.json"),
                    help="the tagger's tag list (bundle vocab.json)")
parser.add_argument("--out", required=True, help="directory that gets tagVocabulary.json and wordTags.json")
args = parser.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)
PLAIN = re.compile(r"[a-z0-9][a-z0-9 \-]*")
key = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())

model = json.load(open(args.modelVocab)); ms = set(model)
old = json.load(open(args.vocabulary)); oldCanon, dropped = old["canonical"], old["dropped"]
oldMembers = {m: name for name, e in oldCanon.items() for m in e["members"]}
searchable = sorted((set(oldMembers) & ms) | {t for t in ms if t not in oldMembers and t not in dropped and PLAIN.fullmatch(t)})

# R1/R2: groups of spelling variants
variants = collections.defaultdict(list)
for t in searchable:
    variants[key(t)].append(t)
def groupName(vs):
    for v in vs:
        if v in oldCanon: return v
    hyph = [v for v in vs if "-" in v]
    return hyph[0] if hyph else vs[0]
nameOfKey = {k: groupName(vs) for k, vs in variants.items()}
groupOfTag = {t: nameOfKey[key(t)] for t in searchable}

def trainFonts(oldName, member):
    return oldCanon[oldName].get("trainFonts", {}).get(member, 0)
def primary(oldName):
    """the member a former group's concept now points at"""
    ms_ = [m for m in oldCanon[oldName]["members"] if m in groupOfTag]
    if oldName in ms_: return oldName
    return max(ms_, key=lambda m: (trainFonts(oldName, m), m)) if ms_ else None
newOfOld = {n: groupOfTag[p] for n in oldCanon if (p := primary(n))}

canonical = {}
for k, vs in variants.items():
    name = nameOfKey[k]
    src = next((oldCanon[oldMembers[v]] for v in vs if v in oldMembers), None)
    canonical[name] = {"facet": src["facet"] if src else "other", "members": vs, "aliases": {v: 1.0 for v in vs}, "implies": [],
                       "trainFonts": {v: trainFonts(oldMembers[v], v) for v in vs if v in oldMembers},
                       "memberAuc": {v: oldCanon[oldMembers[v]]["memberAuc"][v] for v in vs if v in oldMembers and v in oldCanon[oldMembers[v]].get("memberAuc", {})}}

fate = collections.Counter()
def addAlias(phrase, weight, oldName):
    k = key(phrase)
    if k in variants:                                   # R3: the phrase is a tag
        target, w = nameOfKey[k], 1.0; fate["alias is a tag -> that tag"] += 1
    elif oldName in newOfOld:                           # R4
        target, w = newOfOld[oldName], weight; fate["other alias -> primary member of its old group"] += 1
    else:
        fate["alias dropped (old group has no searchable member)"] += 1; return
    a = canonical[target]["aliases"]
    a[phrase] = max(a.get(phrase, 0), w)
for oldName, e in oldCanon.items():
    for phrase, w in e["aliases"].items(): addAlias(phrase, w, oldName)
    if oldName not in e["aliases"]: addAlias(oldName, 1.0, oldName)          # R5
for oldName, e in oldCanon.items():                     # R6
    if oldName in newOfOld:
        c = canonical[newOfOld[oldName]]
        for parent in e["implies"]:
            if parent in newOfOld and newOfOld[parent] != newOfOld[oldName] and newOfOld[parent] not in c["implies"]:
                c["implies"].append(newOfOld[parent])

json.dump({"canonical": dict(sorted(canonical.items())), "dropped": dropped}, open(f"{OUT}/tagVocabulary.json", "w"), indent=1)

# the caption table, mapped onto the new groups
wt = json.load(open(args.wordTags))
moved = kept = lost = 0
for word, entries in wt["words"].items():
    out, seen = [], set()
    for g, *rest in entries:
        if g in canonical: n = g; kept += 1
        elif g in newOfOld: n = newOfOld[g]; moved += 1
        else: lost += 1; continue
        if n not in seen: seen.add(n); out.append([n, *rest])
    wt["words"][word] = out
json.dump(wt, open(f"{OUT}/wordTags.json", "w"), ensure_ascii=False, separators=(",", ":"))

multi = {n: c["members"] for n, c in canonical.items() if len(c["members"]) > 1}
print(f"model tags {len(model)} | searchable {len(searchable)} | groups now {len(canonical)} (was {len(set(groupOfTag.values()) | set())} -> see below)")
print(f"groups holding more than one tag: {len(multi)} (all spelling variants); old canonical groups: {len(oldCanon)}")
print("alias fates:", dict(fate))
print(f"caption table entries: kept {kept}, moved to a primary member {moved}, lost {lost}")
for n in ["horror", "christmas", "sans-serif", "serif", "cute", "condensed", "art-deco"]:
    c = canonical.get(n) or canonical.get(newOfOld.get(n, ""))
    print(f"  {n:11s} -> group {c and [k for k, v in canonical.items() if v is c][0]!r:13} members {c and c['members']} | aliases {c and sorted(c['aliases'])[:8]}")
print("old groups whose name is not a tag, now point at:", {n: newOfOld[n] for n in oldCanon if n not in ms and n in newOfOld})
