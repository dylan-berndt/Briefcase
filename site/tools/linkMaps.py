"""Makes the maps clickable: for each map point, the font page it opens.

The maps (site/frontend/public/maps/*.html) are Plotly exports whose points carry only a hover label, "<family>
<style>", and they cannot be regenerated identically (t-SNE is not deterministic). So the page address of each point
goes in a file beside the map instead, <map>.links.json, in the same order as the map's points:

    {"count": 21179, "labels": "<sha1 of the labels>", "slugs": ["abel", null, ...]}

The map page looks a clicked point up by its number. null is a point with no font page (the map holds fonts the search
does not list). Run it again after the bundle changes, since a slug can change; --check only reports.

    python site/tools/linkMaps.py                  # every map
    python site/tools/linkMaps.py --check
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
sys.path.append(os.path.join(SITE, "backend"))
from slugs import LabelMatcher, assignSlugs  # noqa: E402

# which sources a map was drawn from; a map not listed here may hold any
MAP_SOURCES = {"flower": {"google"}, "routes": {"google", "dafont"}}


def mapLabels(html):
    """The hover labels of the map's points, from the data its Plotly.newPlot call is given."""
    start = html.rfind("Plotly.newPlot(")
    if start < 0:
        raise ValueError("no Plotly.newPlot call")
    decoder = json.JSONDecoder()
    position = html.index(",", start)                      # after the graph div's id
    while html[position] in ", \n\t":
        position += 1
    traces, _ = decoder.raw_decode(html, position)
    labelled = [t for t in traces if isinstance(t.get("text"), list)]
    if len(labelled) != 1:
        raise ValueError(f"expected one trace with labels, found {len(labelled)}")
    return labelled[0]["text"]


def linkMap(labels, fonts, slugs, sources=None):
    pool = [i for i, f in enumerate(fonts) if sources is None or f["source"] in sources]
    matcher = LabelMatcher([fonts[i] for i in pool])
    out = []
    for label in labels:
        j = matcher.match(label)
        out.append(slugs[pool[j]] if j is not None else None)
    return out


def labelsHash(labels):
    return hashlib.sha1("\n".join(labels).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=os.path.join(SITE, "frontend", "public", "maps"))
    parser.add_argument("--fonts", default=os.path.join(SITE, "backend", "data", "fonts.json"))
    parser.add_argument("--check", action="store_true", help="report only; exit 1 if a links file is missing or stale")
    args = parser.parse_args()

    with open(args.fonts, encoding="utf-8") as file:
        fonts = json.load(file)
    slugs = assignSlugs(fonts)

    stale = False
    for name in sorted(os.listdir(args.maps)):
        if not name.endswith(".html"):
            continue
        stem = name[:-5]
        with open(os.path.join(args.maps, name), encoding="utf-8") as file:
            labels = mapLabels(file.read())
        links = linkMap(labels, fonts, slugs, MAP_SOURCES.get(stem))
        linked = sum(slug is not None for slug in links)
        result = {"count": len(labels), "labels": labelsHash(labels), "slugs": links}
        path = os.path.join(args.maps, stem + ".links.json")
        print(f"{stem}: {linked}/{len(labels)} points link to a font page ({len(set(s for s in links if s))} fonts)")
        text = json.dumps(result, separators=(",", ":"))
        if args.check:
            if not os.path.exists(path) or open(path, encoding="utf-8").read() != text:
                print(f"  {stem}.links.json is missing or out of date")
                stale = True
        else:
            with open(path, "w", encoding="utf-8") as file:
                file.write(text)
    if stale:
        sys.exit(1)


if __name__ == "__main__":
    main()
