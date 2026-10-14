"""URL slugs for the font pages, and the matching of map point labels onto them.

A font page is /font/<slug>, one per family (a bundle entry). The slug is the family name folded to ASCII. Two entries
that fold to the same slug (a Google and a DaFont font of one name) both get their source appended, and any that still
collide get a number, in key order, so the result depends only on the bundle.

The maps label each point "<family> <style>" (the font file's name); matchLabel finds the bundle family such a label
belongs to.
"""

import hashlib
import re
import unicodedata
from collections import defaultdict


def slugify(text):
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")


def assignSlugs(fonts):
    """One slug per font, in the order of `fonts`."""
    bases = []
    for font in fonts:
        # a name with no ASCII letters at all (Arabic, Thai...) still needs a stable address
        bases.append(slugify(font["name"]) or "font-" + hashlib.sha1(font["key"].encode()).hexdigest()[:8])
    groups = defaultdict(list)
    for i, base in enumerate(bases):
        groups[base].append(i)

    slugs = [None] * len(fonts)
    taken = set()
    for base, members in groups.items():
        if len(members) == 1:
            slugs[members[0]] = base
            taken.add(base)
    for base, members in groups.items():
        if len(members) == 1:
            continue
        for i in sorted(members, key=lambda i: fonts[i]["key"]):
            slug = f"{base}-{slugify(fonts[i]['source'])}"
            n = 2
            while slug in taken:
                slug = f"{base}-{slugify(fonts[i]['source'])}-{n}"
                n += 1
            slugs[i] = slug
            taken.add(slug)
    return slugs


STYLE_WORDS = {"regular", "normal", "bold", "italic", "oblique", "light", "medium", "book", "thin", "black", "heavy",
               "semi", "demi", "extra", "ultra", "condensed", "expanded", "narrow", "wide", "roman", "reg", "std",
               "semibold", "demibold", "extrabold", "extralight", "bolditalic"}
SHORT_NAME = 6   # letters; a shorter family name has to be followed by nothing but style words to count


def words(text):
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()).split()


class LabelMatcher:
    """Maps a map label ("Abel Regular", "!Paul Maul Bold") to the index of its family in `fonts`, or None.

    The longest leading run of the label's words that spells a family name wins ("Abhaya Libre Bold" is Abhaya Libre,
    not Abhaya). Spelling is compared with spaces ignored as well, as the label and the family disagree on them
    ("!Paul Maul" / "!PaulMaul"). Where a Google and a DaFont family share a name the Google one is used."""

    def __init__(self, fonts, prefer="google"):
        self.exact = {}
        self.squashed = {}
        order = sorted(range(len(fonts)), key=lambda i: (fonts[i]["source"] != prefer, fonts[i]["key"]))
        for i in reversed(order):  # reversed, so the preferred family is written last and wins
            w = words(fonts[i]["name"])
            if not w:
                continue
            self.exact[" ".join(w)] = i
            self.squashed["".join(w)] = i
        self.longest = max((len(k.split()) for k in self.exact), default=0)

    def match(self, label):
        w = words(label)
        for n in range(min(len(w), self.longest), 0, -1):
            i = self.exact.get(" ".join(w[:n]))
            if i is not None:
                # "3D Animals Regular" is not the family "3D"
                if len("".join(w[:n])) < SHORT_NAME and not all(x in STYLE_WORDS or x.isdigit() for x in w[n:]):
                    return None
                return i
        # spelled with different spacing: the first n words, run together, may be a family name run together
        joined = ""
        found = None
        for n in range(1, len(w) + 1):
            joined += w[n - 1]
            if joined in self.squashed:
                found = self.squashed[joined]
        return found
