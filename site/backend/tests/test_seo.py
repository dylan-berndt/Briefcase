import json
import os
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from urllib.parse import quote, unquote

import numpy as np
import pytest

from seo import CLOSE_SHOWN, TAGS_MAX, FontProfiles
from slugs import LabelMatcher, assignSlugs, cleanName, fontPath, specimenPath, words

SITEMAP = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def font(name, source="google"):
    return {"key": f"{source}:{name}", "name": name, "source": source}


# ------------------------------------------------------------ slugs


def test_a_slug_is_the_name_as_written():
    assert assignSlugs([font("Abril Fatface"), font("!Basket of Hammers"), font("Åkerbär & Co.")]) == \
        ["Abril Fatface", "!Basket of Hammers", "Åkerbär & Co."]


def test_addresses_encode_every_reserved_character():
    assert fontPath("Abril Fatface") == "/font/Abril%20Fatface"
    assert fontPath("13/5Atom Sans") == "/font/13%2F5Atom%20Sans"
    assert fontPath("What?#&") == "/font/What%3F%23%26"
    assert specimenPath("Abel") == "/specimen/Abel.webp"


def test_slugs_are_unique_ignoring_case_and_stable():
    fonts = [font("Abel"), font("Abel", "dafont"), font("ABEL", "dafont2"), font("Abel Two"), font("..")]
    slugs = assignSlugs(fonts)
    assert len({s.lower() for s in slugs}) == len(fonts)
    assert slugs[3] == "Abel Two"
    assert slugs[0] == "Abel (Google Fonts)" and slugs[1] == "Abel (DaFont)"        # a shared name gets its source
    assert slugs[4] not in ("..", ".")
    assert assignSlugs(fonts) == slugs and assignSlugs(list(reversed(fonts)))[::-1] == slugs


def test_a_name_is_cleaned_of_control_characters_and_runs_of_space():
    assert assignSlugs([font("  Spaced \t  Out\n")]) == ["Spaced Out"]


# ------------------------------------------------------------ map labels


def test_labels_match_the_longest_family_name():
    fonts = [font("Abhaya"), font("Abhaya Libre"), font("!PaulMaul", "dafont"), font("Abel"), font("Abel", "dafont")]
    matcher = LabelMatcher(fonts)
    assert matcher.match("Abhaya Libre Bold") == 1
    assert matcher.match("Abhaya Regular") == 0
    assert matcher.match("!Paul Maul Bold") == 2              # the label spaces the name differently
    assert matcher.match("Abel Regular") == 3                 # Google wins a shared name
    assert matcher.match("Unknown Regular") is None
    assert matcher.match("") is None


def test_a_short_family_name_needs_only_style_words_after_it():
    matcher = LabelMatcher([font("3D"), font("Abel")])
    assert matcher.match("3D Bold Italic") == 0
    assert matcher.match("3D Animals Regular") is None


# ------------------------------------------------------------ pages


def test_font_page_is_complete_html_without_javascript(client, app):
    bundle = app.extensions["bundle"]
    name = bundle.fonts[7]["name"]
    response = client.get(fontPath(bundle.slugs[7]))
    html = response.get_data(as_text=True)
    assert response.status_code == 200 and "text/html" in response.content_type
    assert f"<title>{name} font" in html and html.count("<title>") == 1       # the shell's own title is replaced
    assert html.count('name="description"') == 1 and 'content="old"' not in html
    assert f'<link rel="canonical" href="https://font-search.com{fontPath(name)}">' in html
    assert "<noscript>" not in html
    assert re.search(r'<div id="root">\s*<main class="Seo">', html)             # content is inside the app's root
    assert bundle.fonts[7]["url"] in html and f"{name}\n" in html
    assert f'src="{specimenPath(name)}"' in html
    assert html.count("application/ld+json") >= 1


def test_the_title_links_to_where_the_font_is_hosted(client, app):
    bundle = app.extensions["bundle"]
    html = client.get(fontPath(bundle.slugs[7])).get_data(as_text=True)
    h1 = re.search(r"<h1>(.*?)</h1>", html, re.DOTALL).group(1)
    assert f'href="{bundle.fonts[7]["url"]}"' in h1 and "<svg" in h1


def test_font_page_lists_tags_close_tags_and_similar_fonts(client, app):
    bundle = app.extensions["bundle"]
    html = client.get(fontPath(bundle.slugs[10])).get_data(as_text=True)
    assert "<h2>Tags</h2>" in html
    assert re.search(r'href="/\?q=[a-z%0-9\-]+"', html)
    similar = html.split("Similar fonts")[1].split("About Font Search")[0]
    related = set(re.findall(r'href="/font/([^"]+)"', similar))
    assert len(related) == 6 and quote(bundle.slugs[10], safe="") not in related


def test_unknown_font_is_a_404_that_is_not_indexed(client):
    response = client.get("/font/Not%20A%20Real%20Font")
    assert response.status_code == 404
    assert 'name="robots" content="noindex"' in response.get_data(as_text=True)


def test_the_name_in_the_wrong_case_redirects(client, app):
    slug = app.extensions["bundle"].slugs[3]
    response = client.get("/font/" + quote(slug.upper(), safe=""))
    assert response.status_code == 301 and response.headers["Location"].endswith(fontPath(slug))


def test_font_info_matches_the_page(client, app):
    bundle = app.extensions["bundle"]
    info = client.get("/api/font/info/" + quote(bundle.slugs[4], safe="")).json
    assert info["key"] == bundle.fonts[4]["key"] and info["tags"] and len(info["related"]) == 6
    assert set(info) >= {"closeTags", "about", "path", "specimen", "title"}
    assert client.get("/api/font/info/nope").status_code == 404


def test_names_with_a_slash_work(makeApp, tmp_path):
    # the fake bundle's names have none, so check the routes accept the encoded form directly
    app = makeApp()
    bundle = app.extensions["bundle"]
    client = app.test_client()
    bundle.fonts[5]["name"] = "13/5 Atom"
    bundle.slugs[5] = "13/5 Atom"
    bundle.indexBySlug["13/5 Atom"] = 5
    app.extensions["pages"].page.cache_clear()
    assert client.get("/font/13%2F5%20Atom").status_code == 200
    assert client.get("/specimen/13%2F5%20Atom.webp").status_code == 200
    assert client.get("/api/font/info/13%2F5%20Atom").json["name"] == "13/5 Atom"


def test_specimen_has_an_address_that_does_not_change(client, app):
    bundle = app.extensions["bundle"]
    response = client.get(specimenPath(bundle.slugs[2]))
    assert response.status_code == 200 and response.mimetype == "image/webp"
    assert response.data == bundle.specimen(2)
    assert client.get("/specimen/nope.webp").status_code == 404


def test_site_pages_describe_the_site_for_crawlers(client):
    home = client.get("/").get_data(as_text=True)
    assert home.count("<title>") == 1
    assert '"SearchAction"' in home
    assert 'rel="canonical"' in home and "https://font-search.com/" in home
    assert len(re.findall(r'href="/font/', home)) >= 40                  # links a crawler can follow in
    for path in ("/map", "/about"):
        page = client.get(path).get_data(as_text=True)
        assert page.count("<title>") == 1 and f'rel="canonical" href="https://font-search.com{path}"' in page
    other = client.get("/some/client/route").get_data(as_text=True)
    assert 'id="root"' in other and 'rel="canonical" href="https://font-search.com/"' in other


def test_site_url_is_configurable(makeApp):
    html = makeApp(SITE_URL="https://example.test/").test_client().get("/").get_data(as_text=True)
    assert 'href="https://example.test/"' in html


# ------------------------------------------------------------ sitemap


def test_sitemap_lists_every_font_once(client, app):
    bundle = app.extensions["bundle"]
    response = client.get("/sitemap.xml")
    assert response.status_code == 200 and "xml" in response.content_type
    locs = [e.text for e in ET.fromstring(response.data).iter(SITEMAP + "loc")]
    assert locs[:3] == ["https://font-search.com/", "https://font-search.com/map", "https://font-search.com/about"]
    fonts = [loc for loc in locs if "/font/" in loc]
    assert sorted(fonts) == sorted(f"https://font-search.com{fontPath(s)}" for s in bundle.slugs)
    assert len(set(locs)) == len(locs)
    assert b"<lastmod>" in response.data and b"image:loc" in response.data
    assert "https://font-search.com/font/Fake%200001" in fonts                   # the name, percent-encoded


# ------------------------------------------------------------ the real names and the real bundle


REAL_FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "fonts.json")


def realFonts():
    try:
        with open(REAL_FONTS, encoding="utf-8") as file:
            text = file.read()
    except OSError:
        pytest.skip("no bundle in site/backend/data")
    if text.startswith("version https://git-lfs"):
        pytest.skip("the bundle is an unpulled git-lfs pointer")
    return json.loads(text)


def test_every_real_font_name_makes_a_usable_address():
    fonts = realFonts()
    slugs = assignSlugs(fonts)
    assert len({s.lower() for s in slugs}) == len(fonts)
    for font, slug in zip(fonts, slugs):
        name = cleanName(font["name"])
        assert slug == name or slug.startswith(name + " (")        # the name, unless told apart from a twin
        assert slug.strip(".") and slug == slug.strip()
        assert not any(unicodedata.category(c)[0] == "C" for c in slug)
        path = fontPath(slug)[len("/font/"):]
        assert not set("/?#") & set(path)                          # nothing reserved is left in the address
        assert unquote(path) == slug                               # and it decodes back to the name


def test_a_sample_of_real_fonts_match_their_own_map_label():
    fonts = realFonts()
    matcher = LabelMatcher(fonts)
    for i in random.Random(0).sample(range(len(fonts)), 300):
        if not words(fonts[i]["name"]):
            continue                                               # a name with no Latin letters has no label to match
        found = matcher.match(fonts[i]["name"] + " Regular")
        assert found is not None and words(fonts[found]["name"]) == words(fonts[i]["name"]), fonts[i]["name"]


def test_the_tag_lists_follow_how_sure_the_model_is_and_stay_short():
    groups = 1300
    profiles = object.__new__(FontProfiles)
    profiles.names = [f"tag{i}" for i in range(groups)]
    sure = np.full(groups, -6.0)
    sure[0] = 8.0                                     # one tag far above the rest
    flat = np.full(groups, 2.0)                       # nothing stands out
    profiles.logit = np.stack([sure, flat]).astype(np.float16)
    tags, close = profiles.tagLists(0)
    assert tags == ["tag0"] and len(close) == CLOSE_SHOWN
    tags, close = profiles.tagLists(1)
    assert len(tags) == TAGS_MAX and len(close) == CLOSE_SHOWN     # capped, never hundreds
    assert not set(tags) & set(close)
