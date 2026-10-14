import re
import xml.etree.ElementTree as ET

from slugs import LabelMatcher, assignSlugs, slugify

SITEMAP = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def font(name, source="google"):
    return {"key": f"{source}:{name}", "name": name, "source": source}


# ------------------------------------------------------------ slugs


def test_slugify_folds_to_ascii():
    assert slugify("Abril Fatface") == "abril-fatface"
    assert slugify("!Basket of Hammers") == "basket-of-hammers"
    assert slugify("Åkerbär  & Co.") == "akerbar-co"
    assert slugify("あい") == ""


def test_slugs_are_unique_and_stable():
    fonts = [font("Abel"), font("Abel", "dafont"), font("Abel Two"), font("あい"), font("Aboreto")]
    slugs = assignSlugs(fonts)
    assert len(set(slugs)) == len(fonts)
    assert slugs[2] == "abel-two" and slugs[4] == "aboreto"
    assert slugs[0] == "abel-google" and slugs[1] == "abel-dafont"      # a shared name gets its source on both
    assert slugs[3].startswith("font-")
    assert assignSlugs(fonts) == slugs and assignSlugs(list(reversed(fonts)))[::-1] == slugs


def test_slug_collisions_that_survive_the_source_get_numbers():
    fonts = [font("A b"), font("A-b"), font("A  b")]
    slugs = assignSlugs(fonts)
    assert len(set(slugs)) == 3


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


# ------------------------------------------------------------ pages


def test_font_page_is_complete_html_without_javascript(client, app):
    bundle = app.extensions["bundle"]
    slug = bundle.slugs[7]
    name = bundle.fonts[7]["name"]
    response = client.get(f"/font/{slug}")
    html = response.get_data(as_text=True)
    assert response.status_code == 200 and "text/html" in response.content_type
    assert f"<title>{name} font" in html and html.count("<title>") == 1       # the shell's own title is replaced
    assert html.count('name="description"') == 1 and 'content="old"' not in html
    assert f'<link rel="canonical" href="https://font-search.com/font/{slug}">' in html
    assert "<noscript>" not in html
    assert re.search(r'<div id="root">\s*<main class="Seo">', html)             # content is inside the app's root
    assert f"<h1>{name}</h1>" in html
    assert f"/specimen/{slug}.webp" in html
    assert "About Font Search" in html                                            # the site is described on every page
    assert html.count("application/ld+json") >= 1
    assert bundle.fonts[7]["url"] in html


def test_font_page_links_to_tags_and_to_other_fonts(client, app):
    bundle = app.extensions["bundle"]
    html = client.get(f"/font/{bundle.slugs[10]}").get_data(as_text=True)
    assert re.search(r'href="/\?q=[a-z%0-9\-]+"', html)
    related = set(re.findall(r'href="/font/([a-z0-9\-]+)"', html))
    assert len(related) >= 6 and bundle.slugs[10] not in related and related <= set(bundle.slugs)


def test_unknown_font_is_a_404_that_is_not_indexed(client):
    response = client.get("/font/not-a-real-font")
    assert response.status_code == 404
    assert 'name="robots" content="noindex"' in response.get_data(as_text=True)


def test_slug_in_the_wrong_case_redirects(client, app):
    slug = app.extensions["bundle"].slugs[3]
    response = client.get(f"/font/{slug.upper()}")
    assert response.status_code == 301 and response.headers["Location"].endswith(f"/font/{slug}")


def test_font_info_matches_the_page(client, app):
    bundle = app.extensions["bundle"]
    info = client.get(f"/api/font/info/{bundle.slugs[4]}").json
    assert info["key"] == bundle.fonts[4]["key"] and info["tags"] and len(info["related"]) == 6
    assert info["previous"]["slug"] == bundle.slugs[3] and info["next"]["slug"] == bundle.slugs[5]
    assert client.get("/api/font/info/nope").status_code == 404


def test_specimen_has_an_address_that_does_not_change(client, app):
    bundle = app.extensions["bundle"]
    response = client.get(f"/specimen/{bundle.slugs[2]}.webp")
    assert response.status_code == 200 and response.mimetype == "image/webp"
    assert response.data == bundle.specimen(2)
    assert client.get("/specimen/nope.webp").status_code == 404


def test_site_pages_describe_the_site_for_crawlers(client):
    home = client.get("/").get_data(as_text=True)
    assert "<title>Font Search – find free fonts" in home
    assert '"@type": "SearchAction"' in home or '"SearchAction"' in home
    assert 'rel="canonical"' in home and "https://font-search.com/" in home
    assert len(re.findall(r'href="/font/', home)) >= 40                  # links a crawler can follow in
    assert "<title>Font Maps" in client.get("/map").get_data(as_text=True)
    assert "<title>About Font Search" in client.get("/about").get_data(as_text=True)
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
    assert sorted(fonts) == sorted(f"https://font-search.com/font/{s}" for s in bundle.slugs)
    assert len(set(locs)) == len(locs)
    assert b"<lastmod>" in response.data and b"image:loc" in response.data
