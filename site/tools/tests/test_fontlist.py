import json

import pandas as pd

from fontList import listFonts, preferredFile
from synthFonts import makeFont, LOWER, UPPER, DIGITS


def metadata(folder, name, category="SANS_SERIF"):
    (folder / "METADATA.pb").write_text(f'name: "{name}"\ncategory: "{category}"\n')


def buildGoogle(root):
    fonts = root / "google" / "fonts"
    for license, folder, name, chars, category in [
        ("ofl", "alpha", "Alpha Sans", LOWER + UPPER, "SANS_SERIF"),
        ("apache", "space", "Space Word", LOWER, "DISPLAY"),
        ("ofl", "icons", "Material Icons", LOWER, "ICONS"),
        ("ofl", "symbolic", "Fancy Symbols", LOWER, "DISPLAY"),
        ("ofl", "caps", "Only Caps", UPPER, "DISPLAY"),
    ]:
        directory = fonts / license / folder
        directory.mkdir(parents=True)
        makeFont(directory / f"{name.replace(' ', '')}-Regular.ttf", chars)
        metadata(directory, name, category)
    return str(fonts)


def buildDafont(root):
    base = root / "dafont"
    (base / "fonts" / "a").mkdir(parents=True)
    (base / "cache").mkdir()
    makeFont(base / "fonts" / "a" / "Wobbly-Regular.ttf", LOWER)
    makeFont(base / "fonts" / "a" / "Wobbly-Bold.ttf", LOWER)
    makeFont(base / "fonts" / "a" / "Doodads.ttf", LOWER)
    makeFont(base / "fonts" / "a" / "Orphan.ttf", LOWER)
    pd.DataFrame({
        "base_font_name": ["Wobbly", "Wobbly", "Doodads", "Orphan"],
        "filename": ["Wobbly-Bold.ttf", "a/Wobbly-Regular.ttf", "Doodads.ttf", "Orphan.ttf"],
        "category": ["Fancy", "Fancy", "Dingbats", "Fancy"],
        "theme": ["Various"] * 4,
    }).to_csv(base / "info.csv", index=False)
    (base / "cache" / "font_list.json").write_text(json.dumps([
        {"name": "Wobbly", "url": "https://www.dafont.com/wobbly.font", "author": "Ann"},
        {"name": "Doodads", "url": "https://www.dafont.com/doodads.font", "author": "Bo"},
    ]))
    return str(base), str(base / "cache" / "font_list.json")


def test_google_and_dafont_lists(tmp_path):
    google = buildGoogle(tmp_path)
    dafont, pages = buildDafont(tmp_path)
    fonts, skipped = listFonts(google, dafont, pages)
    by = {f["key"]: f for f in fonts}

    assert set(by) == {"google:Alpha Sans", "google:Space Word", "dafont:Wobbly"}
    # icon / symbol fonts and dingbats are gone, missing-lowercase fonts too, a family without a page is not listed
    assert not any(k in by for k in ("google:Material Icons", "google:Fancy Symbols", "google:Only Caps",
                                     "dafont:Doodads", "dafont:Orphan"))
    assert skipped == {"dingbat or icon font": 3, "no lowercase a-z": 1, "no page to link to": 1}

    alpha = by["google:Alpha Sans"]
    assert alpha["url"] == "https://fonts.google.com/specimen/Alpha+Sans" and alpha["creator"] is None
    wobbly = by["dafont:Wobbly"]
    assert wobbly["url"] == "https://www.dafont.com/wobbly.font" and wobbly["creator"] == "Ann"
    assert wobbly["path"].endswith("Wobbly-Regular.ttf")  # the regular style represents the family
    assert len(by) == len(fonts)  # keys unique


def test_family_keys_are_stable_across_runs(tmp_path):
    google = buildGoogle(tmp_path)
    first, _ = listFonts(google, None, None)
    second, _ = listFonts(google, None, None)
    assert [f["key"] for f in first] == [f["key"] for f in second]


def test_preferred_file():
    assert preferredFile(["x/A-Bold.ttf", "x/A-Regular.ttf"]).endswith("A-Regular.ttf")
    assert preferredFile(["x/A[wght].ttf", "x/A-Italic[wght].ttf"]).endswith("A[wght].ttf")
    assert preferredFile(["x/Long-Name-Style.ttf", "x/Short.ttf"]).endswith("Short.ttf")


def test_no_sources(tmp_path):
    assert listFonts(None, None, None) == ([], {})
