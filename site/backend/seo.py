"""What a crawler (or a visitor without JavaScript) sees: a page per font, and a summary of the site in every page.

The React app is a single page whose built index.html has an empty #root, so a crawler that does not run scripts sees
nothing. Here the server fills that shell in, per address: a <title>, description, canonical address, social card and
JSON-LD go into the <head>, and plain HTML goes inside #root. React replaces #root's contents when it starts, so none of
this is hydrated or has to match what React draws. Nothing is generated ahead of time or written to disk: the pages are
templates (templates/) filled from the bundle.

FontProfiles is the one expensive piece: every font's tag probabilities, kept so a page can name the tags it has and the
ones close to it. The fonts a page lists as similar are the first results of the search its "more like this" link opens
(the page's own tags as the query), so the two always agree.
"""

import os
import re
from functools import lru_cache

import numpy as np

from slugs import fontPath, specimenPath

SOURCE_LABELS = {"google": "Google Fonts", "dafont": "DaFont"}
TAG_FLOOR = 0.3          # a font "has" a tag above this probability
CLOSE_FLOOR = 0.1        # and a tag between this and TAG_FLOOR is close to being one of its tags
RELATED = 6
TAGS_SHOWN = 12
CLOSE_SHOWN = 10


def sourceLabel(source):
    return SOURCE_LABELS.get(source, source)


def tagPhrase(tags):
    """["a", "b", "c"] -> "a, b and c"."""
    if len(tags) <= 1:
        return "".join(tags)
    return ", ".join(tags[:-1]) + " and " + tags[-1]


class FontProfiles:
    """Per font, the probability of each tag group, as bytes (12k fonts x 1.8k groups is 23 MB)."""

    def __init__(self, index, chunk=1024):
        self.names = index.groupNames
        n, g = index.numFonts, len(self.names)
        self.probability = np.zeros((n, g), dtype=np.uint8)
        for start in range(0, n, chunk):
            ids = np.arange(start, min(n, start + chunk))
            block = index.groupProbabilities(ids)                     # [groups, len(ids)]
            self.probability[ids] = np.round(block.T * 255).astype(np.uint8)
        probability = self.probability.astype(np.float32) / 255
        self.mean = probability.mean(axis=0)
        self.std = probability.std(axis=0) + 1e-3

    def tagLists(self, i):
        """(tags, close): the tags the font has, most likely first, and the ones just short of that, most telling first
        (a tag's z-score is how far above its usual probability across the corpus this font puts it)."""
        p = self.probability[i].astype(np.float32) / 255
        has = [g for g in np.argsort(-p, kind="stable") if p[g] >= TAG_FLOOR][:TAGS_SHOWN]
        if not has:
            has = list(np.argsort(-p, kind="stable")[:3])
        z = (p - self.mean) / self.std
        taken = set(has)
        near = [g for g in np.argsort(-z, kind="stable") if CLOSE_FLOOR <= p[g] < TAG_FLOOR and g not in taken]
        return [self.names[g] for g in has], [self.names[g] for g in near[:CLOSE_SHOWN]]

class Shell:
    """The built index.html, with a page's head and body put into it."""

    def __init__(self, staticDir):
        self.path = os.path.join(staticDir, "index.html")
        self.stamp = None
        self.text = ""

    def load(self):
        try:
            stamp = os.stat(self.path).st_mtime_ns
        except FileNotFoundError:  # no frontend build next to the server (development): a bare page
            return '<!doctype html><html><head><meta charset="utf-8"></head><body><div id="root"></div></body></html>'
        if stamp != self.stamp:  # re-read after a new build is copied in, without a restart
            with open(self.path, encoding="utf-8") as file:
                self.text = file.read()
            self.stamp = stamp
        return self.text

    def render(self, head, body):
        html = self.load()
        html = re.sub(r"<title>.*?</title>", "", html, flags=re.DOTALL)
        html = re.sub(r'<meta\s+name="description"[^>]*>', "", html)
        html = re.sub(r"<noscript>.*?</noscript>", "", html, flags=re.DOTALL)   # the body below says more
        html = html.replace("</head>", head + "</head>", 1) if "</head>" in html else head + html
        root = '<div id="root"></div>'
        return html.replace(root, f'<div id="root">{body}</div>', 1) if root in html else html + body


def aboutParagraphs(summary):
    """The description of the site shown on every page; also what the font page tells a visitor."""
    return [
        f"Font Search is a free tool for finding fonts by describing them. Type what you want, like “spooky "
        f"handwritten” or “clean geometric sans”, and it ranks {summary['fonts']} free fonts from Google Fonts and "
        f"DaFont against your words. Your words are matched to style tags that a model predicted for every font, and "
        f"you can approve or disapprove results to help correct them.",
        "Fonts are not hosted here. Every result links to the font’s own page on Google Fonts or DaFont, where its "
        "license and download are.",
    ]


class Pages:
    """Builds the data behind each page; app.py renders it."""

    def __init__(self, bundle, index, siteURL):
        self.bundle = bundle
        self.siteURL = siteURL.rstrip("/")
        self.index = index
        self.profiles = FontProfiles(index)
        self.page = lru_cache(maxsize=4096)(self.buildPage)

    def card(self, i):
        font = self.bundle.fonts[i]
        slug = self.bundle.slugs[i]
        return {"slug": slug, "name": font["name"], "source": sourceLabel(font["source"]),
                "path": fontPath(slug), "specimen": specimenPath(slug)}

    def similar(self, i, query):
        """The first results of searching for `query`, which is what the page's "more like this" link opens, without the
        font itself."""
        terms, _, _ = self.index.describe(query, suggest=False)
        return [int(j) for j in self.index.rank(terms)[:RELATED + 1] if j != i][:RELATED]

    def buildPage(self, i):
        bundle = self.bundle
        font = bundle.fonts[i]
        slug = bundle.slugs[i]
        tags, close = self.profiles.tagLists(i)
        query = " ".join(tags)
        label = sourceLabel(font["source"])
        by = f" by {font['creator']}" if font.get("creator") else ""
        look = f" It reads as {tagPhrase(tags[:4])}." if tags else ""
        description = (f"{font['name']}{by} is a free font from {label}.{look} See a preview of it and the fonts most "
                      f"like it, or search {len(bundle.fonts):,} free fonts by describing what you want.")
        return {
            "slug": slug,
            "key": font["key"],
            "name": font["name"],
            "source": label,
            "creator": font.get("creator"),
            "url": font["url"],
            "specimen": specimenPath(slug),
            "tags": tags,
            "closeTags": close,
            "query": query,
            "related": [self.card(j) for j in self.similar(i, query)],
            "about": aboutParagraphs(self.siteSummary()),
            "title": f"{font['name']} font – free preview and similar fonts | Font Search",
            "description": description,
            "path": fontPath(slug),
        }

    def siteSummary(self):
        return {"fonts": f"{len(self.bundle.fonts):,}"}

    def featured(self, count=48):
        """An even spread through the alphabet, so the home page links to fonts a crawler can follow."""
        n = len(self.bundle.fonts)
        step = max(1, n // count)
        return [self.card(i) for i in range(0, n, step)][:count]
