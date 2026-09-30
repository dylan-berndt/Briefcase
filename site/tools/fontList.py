"""Which fonts go on the site, and where each one links to: the dafonts-free subset of DaFont plus Google Fonts.

One entry per family. Dingbat and icon fonts are left out (they would show up as results for style queries), as are
fonts that lack the lowercase a-z the tagger reads.

    key      "<source>:<family>", the stable id votes and ratings are filed under
    path     the font file on this machine (only used to build; not shipped)
"""

import glob
import json
import os
import re
from collections import defaultdict
from urllib.parse import quote_plus

import pandas as pd
from fontTools.ttLib import TTFont

LOWERCASE = set("abcdefghijklmnopqrstuvwxyz")
EXCLUDED_CATEGORY = re.compile(r"dingbat|icon|symbol", re.I)
EXCLUDED_NAME = re.compile(r"\b(icons?|symbols?|emoji|dingbats?)\b", re.I)
URL_FIELDS = ("dafont_link", "url", "link", "href", "page")
CREATOR_FIELDS = ("creator", "author", "designer")


def hasLowercase(path):
    try:
        cmap = TTFont(path, lazy=True, fontNumber=0).getBestCmap() or {}
    except Exception:
        return False
    # membership by code point: some cmaps hold codes outside the Unicode range, which chr() rejects
    return all(ord(char) in cmap for char in LOWERCASE)


def preferredFile(paths):
    """The family's regular style: a file named Regular, else the variable font, else the shortest name."""
    def rank(path):
        name = os.path.basename(path).lower()
        return ("regular" not in name, "[" not in name, len(name), name)
    return sorted(paths, key=rank)[0]


def googleFamilies(googleDir):
    """{family: (path, category)} from <googleDir>/*/*/METADATA.pb next to the font files."""
    families = {}
    for metadata in sorted(glob.glob(os.path.join(googleDir, "*", "*", "METADATA.pb"))):
        with open(metadata, encoding="utf-8") as file:
            text = file.read()
        name = re.search(r'^name: "(.*)"', text, re.M)
        if not name:
            continue
        category = " ".join(re.findall(r'^category: "(.*)"', text, re.M))
        folder = os.path.dirname(metadata)
        files = glob.glob(os.path.join(folder, "*.ttf")) + glob.glob(os.path.join(folder, "*.otf"))
        if files:
            families[name.group(1)] = (preferredFile(files), category)
    return families


def dafontFamilies(dafontDir, pageList):
    """{family: (path, category, url, creator)}. info.csv gives a family's files and category; pageList is the
    dafonts-free cache/font_list.json that holds each family's DaFont page and creator."""
    pages = {}
    if pageList:
        with open(pageList, encoding="utf-8") as file:
            listed = json.load(file)
        if isinstance(listed, dict) and isinstance(listed.get("font_info"), list):
            listed = listed["font_info"]       # the real dafonts-free file: {"dataset_name", "date", "font_info": [...]}
        for item in (listed.values() if isinstance(listed, dict) else listed):
            if isinstance(item, dict) and "name" in item:
                pages[item["name"]] = item

    onDisk = defaultdict(list)
    for path in glob.glob(os.path.join(dafontDir, "fonts", "**", "*"), recursive=True):
        if path.lower().endswith((".ttf", ".otf")):
            onDisk[os.path.basename(path).lower()].append(path)

    info = pd.read_csv(os.path.join(dafontDir, "info.csv"), on_bad_lines="skip")
    families = {}
    for name, rows in info.groupby("base_font_name"):
        paths = [p for f in rows["filename"].astype(str) for p in onDisk.get(os.path.basename(f).lower(), [])]
        if not paths:
            continue
        page = pages.get(name, {})
        creator = next((page[k] for k in CREATOR_FIELDS if page.get(k)), None)
        if creator is None and "creator" in rows and rows["creator"].notna().any():
            creator = str(rows["creator"].dropna().iloc[0])      # info.csv carries the creator too
        families[str(name)] = (
            preferredFile(paths),
            " ".join(str(c) for c in set(rows["category"].dropna())),
            next((page[k] for k in URL_FIELDS if page.get(k)), None),
            creator,
        )
    return families


def listFonts(googleDir=None, dafontDir=None, dafontPageList=None):
    """Returns (fonts, skipped): fonts is the list of entries, skipped counts what was left out and why."""
    fonts, skipped = [], defaultdict(int)

    def consider(source, family, path, category, url, creator):
        if EXCLUDED_CATEGORY.search(category) or EXCLUDED_NAME.search(family):
            skipped["dingbat or icon font"] += 1
        elif not hasLowercase(path):
            skipped["no lowercase a-z"] += 1
        elif url is None:
            skipped["no page to link to"] += 1
        else:
            fonts.append({"key": f"{source}:{family}", "name": family, "source": source, "url": url,
                          "creator": creator, "path": path})

    if googleDir:
        for family, (path, category) in sorted(googleFamilies(googleDir).items()):
            consider("google", family, path, category, f"https://fonts.google.com/specimen/{quote_plus(family)}", None)
    if dafontDir:
        for family, (path, category, url, creator) in sorted(dafontFamilies(dafontDir, dafontPageList).items()):
            consider("dafont", family, path, category, url, creator)

    return fonts, dict(skipped)
