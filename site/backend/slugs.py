"""URL slugs for the font pages, and the matching of map point labels onto them.

A font page is /font/<slug>, one per family (a bundle entry). The slug is the family's own name, as it is written
("Abril Fatface"); the address is that name percent-encoded (/font/Abril%20Fatface). Two entries whose names are the
same ignoring case (a Google and a DaFont font of one name) both get their source added, "Abel (Google Fonts)", and any
that still collide get a number, in key order, so the result depends only on the bundle.

The maps label each point "<family> <style>" (the font file's name); LabelMatcher finds the bundle family such a label
belongs to.
"""

import re
import unicodedata
from collections import defaultdict
from urllib.parse import quote

SOURCE_NAMES = {"google": "Google Fonts", "dafont": "DaFont"}


def fontPath(slug):
    """The address of a font's page. Every reserved character is encoded, "/" and "?" included."""
    return "/font/" + quote(slug, safe="")


def specimenPath(slug):
    return "/specimen/" + quote(slug, safe="") + ".webp"


def cleanName(name):
    return re.sub(r"\s+", " ", "".join(c for c in name if unicodedata.category(c)[0] != "C")).strip()


def assignSlugs(fonts):
    """One slug per font, in the order of `fonts`."""
    bases = []
    for font in fonts:
        base = cleanName(font["name"])
        bases.append(base if base.strip(".") else cleanName(font["key"]))   # "." and ".." are not addresses
    groups = defaultdict(list)
    for i, base in enumerate(bases):
        groups[base.lower()].append(i)

    slugs = [None] * len(fonts)
    taken = set()
    for members in groups.values():
        if len(members) == 1:
            slugs[members[0]] = bases[members[0]]
            taken.add(bases[members[0]].lower())
    for members in groups.values():
        if len(members) == 1:
            continue
        for i in sorted(members, key=lambda i: fonts[i]["key"]):
            source = SOURCE_NAMES.get(fonts[i]["source"], fonts[i]["source"])
            slug = f"{bases[i]} ({source})"
            n = 2
            while slug.lower() in taken:
                slug = f"{bases[i]} ({source}) {n}"
                n += 1
            slugs[i] = slug
            taken.add(slug.lower())
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
