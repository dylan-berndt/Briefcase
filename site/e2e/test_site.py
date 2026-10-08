import re
import uuid

import pytest
from playwright.sync_api import expect

from conftest import shot


def search(page, text):
    page.get_by_label("Describe a font").fill(text)
    page.get_by_label("Describe a font").press("Enter")


def names(page):
    return page.locator(".ResultTitle a").all_inner_texts()


def imagesLoaded(page):
    """Every specimen decodes. They are lazy-loaded, so scroll each into view first."""
    page.evaluate("""async () => { for (const img of document.querySelectorAll('.Specimen img')) {
        img.scrollIntoView(); await new Promise(r => setTimeout(r, 30)); } window.scrollTo(0, 0); }""")
    page.wait_for_function("""() => { const imgs = [...document.querySelectorAll('.Specimen img')];
        return imgs.length > 0 && imgs.every(i => i.complete && i.naturalWidth > 0); }""")


def signUp(page, name=None, password="correct horse"):
    name = name or "u" + uuid.uuid4().hex[:10]
    if not page.get_by_label("Username:").is_visible():
        page.get_by_role("button", name="Login").first.click()
    page.get_by_role("button", name="Register").click()
    page.get_by_label("Username:").fill(name)
    page.get_by_label("Password:").fill(password)
    page.get_by_role("button", name="Submit").click()
    expect(page.get_by_role("button", name=name)).to_be_visible()
    return name


def test_search_shows_specimens_that_load(page, site):
    page.goto("/")
    search(page, "zebra stripe")
    expect(page.locator(".ResultWindow")).to_have_count(24)
    imagesLoaded(page)
    expect(page.get_by_label("Tags in your search")).to_have_text("zebra-stripe")
    # 30 fonts have the tag planted, so the whole first page is made of them
    planted = {k.split(":", 1)[1] for k, g in site["planted"].items() if "zebra-stripe" in g}
    assert len(planted) == 30 and set(names(page)) <= planted
    shot(page, "results")
    assert page.errors == []


def test_every_specimen_link_goes_out_in_a_new_tab(page):
    page.goto("/")
    search(page, "serif")
    page.locator(".ResultWindow").first.wait_for()
    links = page.locator(".ResultWindow a")
    for i in range(links.count()):
        assert links.nth(i).get_attribute("target") == "_blank"
        assert "noopener" in links.nth(i).get_attribute("rel")
        assert links.nth(i).get_attribute("href").startswith("https://")


def test_paging_to_the_very_end_and_back(page):
    page.goto("/")
    search(page, "bold")
    expect(page.get_by_text("Page 1 of 13 · 300 fonts")).to_be_visible()
    first = names(page)
    page.get_by_role("button", name="Next").click()
    expect(page.get_by_text("Page 2 of 13")).to_be_visible()
    second = names(page)
    assert not set(first) & set(second)
    assert page.url.endswith("?q=bold&page=2")

    page.get_by_role("navigation").get_by_role("button", name="13", exact=True).click()
    expect(page.get_by_text("Page 13 of 13 · 300 fonts")).to_be_visible()
    expect(page.locator(".ResultWindow")).to_have_count(300 - 12 * 24)
    expect(page.get_by_role("button", name="Next")).to_be_disabled()
    imagesLoaded(page)
    shot(page, "last-page")

    page.go_back()
    expect(page.get_by_text("Page 2 of 13")).to_be_visible()
    assert names(page) == second
    page.go_back()
    expect(page.get_by_text("Page 1 of 13")).to_be_visible()
    assert names(page) == first


def test_reload_keeps_query_and_page(page):
    page.goto("/")
    search(page, "script")
    page.get_by_role("button", name="Next").click()
    expect(page.get_by_text("Page 2 of 13")).to_be_visible()
    before = names(page)
    page.reload()
    expect(page.get_by_text("Page 2 of 13")).to_be_visible()
    assert names(page) == before
    expect(page.get_by_label("Describe a font")).to_have_value("script")


def test_shared_link_past_the_end_lands_on_last_page(page):
    page.goto("/?q=bold&page=500")
    expect(page.get_by_text("Page 13 of 13")).to_be_visible()


def test_scrolling_down_and_back_up_returns_to_the_first_view(page):
    def where():
        return page.evaluate("""() => ({ scrollY: Math.round(scrollY),
            title: Math.round(document.querySelector('.Center > div').getBoundingClientRect().top),
            input: Math.round(document.querySelector('.SearchForm').getBoundingClientRect().top) })""")
    page.goto("/")
    first = where()
    search(page, "bold")
    page.locator(".ResultWindow").first.wait_for()
    assert where() == first  # results appear below, nothing above them moves
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    assert where()["scrollY"] > 1000
    page.evaluate("window.scrollTo(0, 0)")
    assert where() == first


def test_card_has_no_stars_and_name_source_and_thumbs_are_right_aligned(page):
    page.goto("/")
    search(page, "serif")
    card = page.locator(".ResultWindow").first
    assert card.get_by_role("button", name=re.compile("Rate")).count() == 0 and "unrated" not in card.inner_text()
    edge = card.bounding_box()["x"] + card.bounding_box()["width"]
    name, source, thumbs = card.locator(".ResultTitle a"), card.locator(".ResultSource"), card.locator(".Votes")
    for part in (name, source, thumbs):
        box = part.bounding_box()
        assert edge - (box["x"] + box["width"]) < 40          # all three sit against the right edge of the card
    assert name.bounding_box()["y"] < source.bounding_box()["y"] < thumbs.bounding_box()["y"]
    box = card.bounding_box()
    thumbs_box, source_box = thumbs.bounding_box(), source.bounding_box()
    assert box["y"] + box["height"] - (thumbs_box["y"] + thumbs_box["height"]) < 40      # thumbs at the bottom of the card
    assert thumbs_box["y"] - (source_box["y"] + source_box["height"]) >= 16              # with room under the source
    assert box["height"] <= 150                                                          # and the whole result is short
    img = card.locator(".Specimen img").bounding_box()
    assert img["width"] >= 430                                                           # the preview keeps its size...
    assert abs(img["width"] / img["height"] - 640 / 140) < 0.1                           # ...but not the blank bands above and below
    up = card.get_by_role("button", name="This font matched my query")
    assert up.get_attribute("title") == "This font matched my query"
    assert card.get_by_role("button", name="This font did not match my query").get_attribute("title") == "This font did not match my query"
    assert card.get_by_role("button", name="Describe").count() == 0


def test_info_sits_beside_the_preview_on_a_desktop(page):
    page.goto("/")
    search(page, "serif")
    card = page.locator(".ResultWindow").first
    img, info = card.locator(".Specimen").bounding_box(), card.locator(".ResultInfo").bounding_box()
    assert img["x"] + img["width"] <= info["x"] + 1          # to the right of the preview, not under it
    assert abs((img["y"] + img["height"] / 2) - (info["y"] + info["height"] / 2)) < img["height"]  # same row


def test_tag_line_is_one_line_when_it_fits(page):
    page.goto("/")
    search(page, "elegant script not thin")           # there is no negation: "not thin" gives no tag
    tags = page.get_by_label("Tags in your search")
    expect(tags.get_by_role("button", name="elegant: included")).to_be_visible()
    boxes = tags.get_by_role("button").all()
    assert len(boxes) == 2 and tags.get_by_text("thin").count() == 0 and len({round(b.bounding_box()["y"]) for b in boxes}) == 1   # words side by side
    assert tags.get_by_role("listitem").first.bounding_box()["x"] < boxes[0].bounding_box()["x"]  # word, then its box
    assert page.get_by_text("Searching for:").count() == 0       # no label, just the words


def test_tag_line_ticks_are_page_state_and_change_the_results(page):
    page.goto("/")
    requests = []
    page.on("request", lambda r: requests.append(r.url) if "/api/font/" in r.url else None)
    search(page, "elegant script airy slimy")
    tags = page.get_by_label("Tags in your search")
    expect(tags.get_by_role("button", name="elegant: included")).to_be_visible()
    assert page.get_by_label("Tags in your search").locator("[aria-label$=excluded]").count() == 0
    expect(tags.get_by_role("button", name="feminine: included")).to_be_visible()   # guessed for "airy", a plain tag
    suggestion = tags.get_by_role("button", name=re.compile(r": off$")).first         # suggested for "slimy", unticked
    expect(suggestion).to_be_visible()
    text = tags.inner_text()
    assert "→" not in text and "similar" not in text.lower()
    shot(page, "tags")
    assert sum("/api/font/tags" in u for u in requests) == 1

    def results():
        return page.locator(".ResultTitle a").all_inner_texts()

    def settles(condition):
        """The fonts are fetched after the box changes, so wait for the list to satisfy the condition."""
        for _ in range(60):
            if condition(results()):
                return
            page.wait_for_timeout(100)
        raise AssertionError(f"the fonts never settled: {results()[:4]}")

    before = results()
    # tick -> empty -> tick on a typed tag; every step changes the fonts shown
    page.get_by_role("button", name="script: included").click()
    expect(tags.get_by_role("button", name="script: off")).to_be_visible()
    settles(lambda r: r != before)
    off = results()
    assert off != before
    tags.get_by_role("button", name="script: off").click()
    expect(tags.get_by_role("button", name="script: included")).to_be_visible()
    settles(lambda r: r == before)

    # a suggestion's first click ticks it
    name = suggestion.get_attribute("aria-label").removesuffix(": off")
    suggestion.click()
    expect(tags.get_by_role("button", name=f"{name}: included")).to_be_visible()
    settles(lambda r: r != before)

    # none of that was sent to the server as anything but the final list, and the URL stays q/page
    assert sum("/api/font/tags" in u for u in requests) == 1
    assert "tags=" not in page.url and "ignore" not in page.url
    page.reload()                                                 # a reload starts from the query's own tags
    expect(tags.get_by_role("button", name=f"{name}: off")).to_be_visible()


def test_unticking_everything_shows_a_prompt(page):
    page.goto("/")
    search(page, "bold")
    tags = page.get_by_label("Tags in your search")
    tags.get_by_role("button", name="bold: included").click()
    expect(page.get_by_text("Tick a tag to see fonts.")).to_be_visible()
    expect(page.locator(".ResultWindow")).to_have_count(0)


def test_search_bar_has_no_border_rounded_corners_and_a_shadow_below(page):
    page.goto("/")
    style = page.evaluate("""() => { const s = getComputedStyle(document.querySelector('.SearchForm input'));
        return { border: s.borderTopWidth, radius: s.borderTopLeftRadius, shadow: s.boxShadow } }""")
    assert style["border"] == "0px"
    assert float(style["radius"].removesuffix("px")) >= 8
    shadow = style["shadow"]            # "rgba(...) 0px 8px 12px -4px": straight down, no sideways offset
    assert "0px 8px" in shadow
    page.get_by_label("Describe a font").focus()      # focus must not bring a border back
    focused = page.evaluate("""() => { const s = getComputedStyle(document.querySelector('.SearchForm input'));
        return { border: s.borderTopWidth, outline: s.outlineStyle } }""")
    assert focused == {"border": "0px", "outline": "none"}
    shot(page, "searchbar")


@pytest.mark.parametrize("width,column,content", [(1400, 980, 744), (1920, 1344, 1004)])
def test_the_column_is_70_percent_wide_with_the_content_at_its_own_width(browser, site, width, column, content):
    context = browser.new_context(viewport={"width": width, "height": 1000}, base_url=site["url"])
    page = context.new_page()
    page.goto("/")
    search(page, "serif")
    page.locator(".ResultWindow").first.wait_for()
    sizes = page.evaluate("""() => ({ column: document.querySelector('.Shadow').getBoundingClientRect().width,
        results: document.querySelector('.Results').getBoundingClientRect().width })""")
    context.close()
    assert abs(sizes["column"] - column) <= 2     # as wide as it was before it was narrowed
    assert abs(sizes["results"] - content) <= 2   # the results keep the width they have now


def test_the_prompt_is_smaller(page):
    page.goto("/")
    size = page.evaluate("""() => parseFloat(getComputedStyle(document.querySelector('.Center > p')).fontSize)""")
    assert size <= 22     # it was 3vmin, 30px at this viewport


def test_unrecognised_query(page):
    page.goto("/")
    search(page, "qwertyuiop")
    expect(page.get_by_text("No tags recognised in that description.")).to_be_visible()
    expect(page.get_by_text("Not recognised: qwertyuiop.")).to_be_visible()
    expect(page.locator(".ResultWindow")).to_have_count(0)


def test_feedback_needs_login_then_persists(page):
    page.goto("/")
    search(page, "serif")
    card = page.locator(".ResultWindow").first
    card.get_by_role("button", name="This font matched my query").click()
    expect(card.get_by_text("Log in to give feedback")).to_be_visible()
    expect(page.get_by_label("Username:")).to_be_visible()

    name = signUp(page)
    # logging in reloads the results with this user's state
    card = page.locator(".ResultWindow").first
    title = card.locator(".ResultTitle a").inner_text()
    yes = card.get_by_role("button", name="This font matched my query")
    yes.click()
    expect(yes).to_have_attribute("aria-pressed", "true")
    shot(page, "feedback")

    page.reload()
    card = page.locator(".ResultWindow").first
    expect(page.get_by_role("button", name=name)).to_be_visible()  # session survived the reload
    assert card.locator(".ResultTitle a").inner_text() == title
    expect(card.get_by_role("button", name="This font matched my query")).to_have_attribute("aria-pressed", "true")

    # the vote belongs to that query
    search(page, "serif bold")
    other = page.locator(".ResultWindow", has_text=title)
    if other.count():
        expect(other.get_by_role("button", name="This font matched my query")).to_have_attribute("aria-pressed", "false")

    page.get_by_role("button", name=name).click()
    page.get_by_role("button", name="Log out").click()
    expect(page.get_by_role("button", name="Login")).to_be_visible()
    expect(page.locator(".ResultWindow").first.get_by_role("button", name="This font matched my query")).to_have_attribute("aria-pressed", "false")


def test_login_errors_are_shown(page):
    page.goto("/")
    page.get_by_role("button", name="Login").first.click()
    page.get_by_label("Username:").fill("nobody")
    page.get_by_label("Password:").fill("wrong password")
    page.get_by_role("button", name="Submit").click()
    expect(page.get_by_text("Invalid username or password.")).to_be_visible()


def test_no_horizontal_scroll_on_a_phone(browser, site):
    context = browser.new_context(viewport={"width": 390, "height": 800}, base_url=site["url"])
    page = context.new_page()
    page.goto("/")
    search(page, "serif")
    page.locator(".ResultWindow").first.wait_for()
    imagesLoaded(page)
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    shot(page, "phone")
    context.close()
    assert overflow <= 0


def test_about_page_renders_the_markdown_with_contents(page):
    page.goto("/")
    page.get_by_role("link", name="About").click()
    assert page.url.endswith("/about")
    expect(page.get_by_role("heading", level=1).first).to_be_visible()
    nav = page.get_by_role("navigation", name="Table of contents")
    expect(nav).to_be_visible()
    links = nav.get_by_role("link")
    assert links.count() >= 2
    # **bold** is bold and $$maths$$ is typeset (about.md uses both)
    assert page.locator(".AboutText strong").count() > 0
    assert int(page.evaluate("getComputedStyle(document.querySelector('.AboutText strong')).fontWeight")) >= 600
    assert page.locator(".AboutText .katex").count() > 0
    assert page.locator(".AboutText .katex-error").count() == 0
    page.wait_for_function("document.fonts.ready.then(() => [...document.fonts].some(f => f.family.includes('KaTeX') && f.status === 'loaded'))")

    # a contents link scrolls a heading that starts below the fold into view
    target = links.last.get_attribute("href").removeprefix("#")
    assert page.evaluate("id => document.getElementById(id).getBoundingClientRect().top > innerHeight", target)
    links.last.click()
    page.wait_for_function("""id => { const r = document.getElementById(id).getBoundingClientRect();
        return r.top >= -2 && r.bottom <= innerHeight; }""", arg=target)   # a heading can land a fraction of a pixel above 0
    assert "#" not in page.url
    shot(page, "about")
    assert page.errors == []


def test_pages_have_their_own_addresses(page):
    # direct visit and reload: the server hands every page address to the app, the router picks the page
    for path, marker in (("/about", ".AboutBody"), ("/map", ".MapArea")):
        page.goto(path)
        expect(page.locator(marker)).to_be_visible()
        assert page.url.endswith(path)
        page.reload()
        expect(page.locator(marker)).to_be_visible()
    page.goto("/about/")                                  # a trailing slash is the same page
    expect(page.locator(".AboutBody")).to_be_visible()
    page.goto("/no/such/page")                            # unknown addresses land on the search page
    expect(page.get_by_label("Describe a font")).to_be_visible()
    assert page.url.split("/", 3)[3] == ""

    # the header links move between pages and the browser buttons follow
    page.get_by_role("link", name="Maps").click()
    expect(page.locator(".MapArea")).to_be_visible()
    assert page.url.endswith("/map")
    page.go_back()
    expect(page.get_by_label("Describe a font")).to_be_visible()
    page.go_forward()
    expect(page.locator(".MapArea")).to_be_visible()

    # a search keeps its ?q= address and Home does not clear it
    page.get_by_role("link", name="Home").click()
    search(page, "zebra stripe")
    expect(page.locator(".ResultWindow")).to_have_count(24)
    assert "q=zebra+stripe" in page.url
    page.get_by_role("link", name="Home").click()
    assert "q=zebra+stripe" in page.url
    expect(page.locator(".ResultWindow")).to_have_count(24)
    assert page.errors == []


@pytest.mark.parametrize("size", [(390, 844), (360, 640), (768, 1024)])
def test_pages_fit_a_phone_or_tablet_screen(browser, site, size):
    width, height = size
    context = browser.new_context(base_url=site["url"], viewport={"width": width, "height": height},
                                  device_scale_factor=2, is_mobile=True, has_touch=True)
    context.set_default_timeout(8000)
    context.add_init_script("window.requestAnimationFrame = cb => setTimeout(() => cb(performance.now()), 1000);")
    page = context.new_page()
    for path in ("/", "/?q=zebra+stripe", "/about", "/map"):
        page.goto(path)
        if "q=" in path:
            expect(page.locator(".ResultWindow")).to_have_count(24)
        elif path == "/about":
            expect(page.locator(".AboutBody")).to_be_visible()
        page.wait_for_timeout(300)
        # nothing makes the page scroll sideways
        assert page.evaluate("document.documentElement.scrollWidth") <= width, path
        # the header's links and button stay on screen, and are finger-sized on a phone
        for box in page.locator(".Bar a, .Bar button").all():
            rect = box.bounding_box()
            assert rect["x"] >= 0 and rect["x"] + rect["width"] <= width, (path, rect)
            assert rect["height"] >= (34 if width <= 700 else 28), (path, rect)
    page.goto("/")
    # a field under 16px makes a phone's browser zoom in when it is tapped
    assert page.evaluate("parseFloat(getComputedStyle(document.querySelector('input[type=text]')).fontSize)") >= 16 or width > 700
    # the title and the line under it are sized by the screen's width on a phone, not by vmin (which was 23px and 8px)
    sizes = page.evaluate("""() => { const title = document.querySelector('.Center p'), line = document.querySelector('.Center > p');
        return [parseFloat(getComputedStyle(title).fontSize), parseFloat(getComputedStyle(line).fontSize), line.scrollWidth, line.clientWidth]; }""")
    if width <= 700:
        assert sizes[0] >= 28 and sizes[1] >= 13, sizes
        assert sizes[2] <= sizes[3] + 1, sizes                                  # and the line is not cut off
        assert abs(sizes[0] - min(max(28, 0.086 * width), 44)) <= 1.5, sizes    # 34px at 390 wide
    else:
        assert abs(sizes[0] - 0.06 * min(width, height)) <= 1.5, sizes          # the desktop's 6vmin, unchanged
    # the search field and the dark column use the width of the screen
    field = page.get_by_label("Describe a font").bounding_box()
    if width <= 700:
        assert field["width"] >= width * 0.7
    page.goto("/about")
    if width <= 900:
        nav = page.get_by_role("navigation", name="Table of contents").bounding_box()
        text = page.locator(".AboutText").bounding_box()
        assert nav["y"] < text["y"] and nav["height"] <= height * 0.4      # contents first, and not the whole screen
    context.close()


@pytest.mark.parametrize("size,band", [((390, 844), 90), ((360, 640), 68), ((414, 896), 96)])
def test_phones_have_a_slice_of_the_background_pinned_to_the_bottom(browser, site, size, band):
    """90px on a 390x844 phone, in proportion to the screen's height on others; the column is the whole width."""
    width, height = size
    context = browser.new_context(base_url=site["url"], viewport={"width": width, "height": height},
                                  device_scale_factor=2, is_mobile=True, has_touch=True)
    context.set_default_timeout(8000)
    context.add_init_script("window.requestAnimationFrame = cb => setTimeout(() => cb(performance.now()), 1000);")
    page = context.new_page()

    def slice_():
        return page.evaluate("""() => { const r = document.querySelector('.Shader').getBoundingClientRect();
            return {top: r.top, bottom: r.bottom, height: r.height, width: r.width, innerHeight}; }""")

    for path in ("/", "/?q=zebra+stripe", "/about"):
        page.goto(path)
        if "q=" in path:
            expect(page.locator(".ResultWindow")).to_have_count(24)
        elif path == "/about":
            expect(page.locator(".AboutBody")).to_be_visible()
        page.wait_for_timeout(300)
        # the whole width, no strips either side
        column = page.evaluate("document.querySelector('.Shadow').getBoundingClientRect().width")
        assert abs(column - width) <= 1, (path, column)
        # the slice is on the screen's bottom edge, whatever the page's scroll position
        for y in (0, 700, "end"):
            page.evaluate("y => window.scrollTo(0, y === 'end' ? document.documentElement.scrollHeight : y)", y)
            page.wait_for_timeout(150)
            found = slice_()
            assert abs(found["bottom"] - height) <= 1 and abs(found["height"] - band) <= 2, (path, y, found)
            assert found["width"] >= width - 1
        # at the end of the page nothing is hidden under it
        content_bottom = page.evaluate("document.querySelector('.Shadow').getBoundingClientRect().bottom")
        assert content_bottom <= height - band + 2, (path, content_bottom)
    # its shadow falls on the slice, and a strip in the column's colour sits just above it
    edge = page.evaluate("""() => { const shadow = getComputedStyle(document.querySelector('.Shader'), '::after');
        const strip = getComputedStyle(document.querySelector('.Shader'), '::before');
        return [shadow.backgroundImage.startsWith('linear-gradient'), parseFloat(shadow.height),
                strip.height, strip.backgroundColor, getComputedStyle(document.querySelector('.Center, .About')).backgroundColor]; }""")
    assert edge[0] and abs(edge[1] - 0.04 * min(width, height)) <= 1, edge
    assert edge[2] == "8px" and edge[3] == "rgb(24, 25, 29)", edge
    # a short page fills the screen above the slice
    page.goto("/")
    page.wait_for_timeout(300)
    assert abs(page.evaluate("document.querySelector('.Shadow').getBoundingClientRect().bottom") - (height - band)) <= 2
    # with the first result cards scrolled under the slice, the row of pixels just above it is the column's colour all
    # the way across: nothing touches the slice's edge
    from io import BytesIO
    from PIL import Image
    page.goto("/?q=zebra+stripe")
    expect(page.locator(".ResultWindow")).to_have_count(24)
    for y in (900, 1500, 2300):
        page.evaluate("y => window.scrollTo(0, y)", y)
        page.wait_for_timeout(200)
        row = Image.open(BytesIO(page.screenshot(clip={"x": 0, "y": height - band - 4, "width": width, "height": 1}))).convert("RGB")
        off = [p for p in row.getdata() if max(abs(p[0] - 24), abs(p[1] - 25), abs(p[2] - 29)) > 2]
        assert not off, (y, off[:3])
    context.close()

    desktop = browser.new_context(base_url=site["url"], viewport={"width": 1400, "height": 900})
    page = desktop.new_page()
    page.goto("/")
    page.wait_for_timeout(300)
    full = page.evaluate("""() => { const r = document.querySelector('.Shader').getBoundingClientRect();
        return [r.top, r.bottom, r.width, innerWidth, innerHeight, getComputedStyle(document.querySelector('.Shader')).zIndex]; }""")
    assert full[:2] == [0, 900] and full[2] == full[3] and full[5] == "0"      # still the whole screen, behind the page
    assert page.evaluate("document.documentElement.scrollHeight - document.querySelector('.Shadow').getBoundingClientRect().bottom") <= 1
    desktop.close()


def test_sitemap_and_robots_point_crawlers_at_real_pages(site):
    import urllib.request
    import xml.etree.ElementTree as ET
    from urllib.parse import urlparse

    def get(path):
        with urllib.request.urlopen(site["url"] + path) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read().decode()

    status, kind, body = get("/robots.txt")
    assert status == 200 and "Sitemap: https://font-search.com/sitemap.xml" in body
    assert "Disallow: /" not in body.replace("Disallow:\n", "")                  # nothing is blocked

    status, kind, body = get("/sitemap.xml")
    assert status == 200 and "xml" in kind
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [e.text for e in ET.fromstring(body).findall("s:url/s:loc", ns)]
    assert set(locs) == {"https://font-search.com/", "https://font-search.com/map", "https://font-search.com/about"}
    for loc in locs:
        # each listed address is served as the app, with no redirect (the same path on this server)
        path = urlparse(loc).path
        status, kind, page = get(path)
        assert status == 200 and "text/html" in kind and 'id="root"' in page, (path, status, kind)


@pytest.mark.parametrize("width,height", [(390, 844), (1280, 800)])
def test_map_has_no_plotly_margin_and_fits_the_scene(browser, site, width, height):
    ctx = browser.new_context(base_url=site["url"], viewport={"width": width, "height": height})
    page = ctx.new_page()
    page.goto("/map")
    page.select_option("#options", "routes")
    frame = page.frame_locator("iframe[title=mapLocation]")
    frame.locator(".plotly-graph-div canvas").first.wait_for(timeout=60000)
    page.wait_for_function("""() => { const f = document.querySelector('iframe[title=mapLocation]');
        const gd = f.contentDocument && f.contentDocument.querySelector('.plotly-graph-div');
        return gd && gd._fullLayout && gd._fullLayout.margin.l === 0 && gd._briefcaseZoom; }""", timeout=30000)
    info = page.evaluate("""() => { const f = document.querySelector('iframe[title=mapLocation]');
        const gd = f.contentDocument.querySelector('.plotly-graph-div'); const L = gd._fullLayout;
        return {margin: L.margin, w: L.width, frameW: f.clientWidth, zoom: gd._briefcaseZoom}; }""")
    assert all(info["margin"][k] == 0 for k in "lrtb")
    assert info["w"] >= info["frameW"] - 20
    assert info["zoom"] == (1.5 if width < height else 1)
    ctx.close()


def test_shader_fix_flags_can_be_switched_on_from_the_page(browser, site):
    """Experiments for iOS's toolbar are html[data-fix] flags: off by default, and each changes the shader's layer."""
    ctx = browser.new_context(base_url=site["url"], viewport={"width": 390, "height": 844})
    page = ctx.new_page()
    page.goto("/about")
    page.wait_for_selector(".Shader canvas")
    info = """() => { const s = document.querySelector('.Shader'), l = document.querySelector('.ShaderLayer');
        return {flag: document.documentElement.dataset.fix || '', op: getComputedStyle(s).opacity,
                bg: getComputedStyle(s).backgroundColor, layerH: Math.round(l.getBoundingClientRect().height),
                shaderH: Math.round(s.getBoundingClientRect().height)}; }"""
    base = page.evaluate(info)
    assert base["flag"] == "" and base["op"] == "1" and base["bg"] == "rgba(0, 0, 0, 0)"
    assert base["layerH"] == base["shaderH"]
    page.evaluate("document.documentElement.dataset.fix = 'bleed op'")
    on = page.evaluate(info)
    assert on["op"] == "0.99" and on["layerH"] == on["shaderH"] + 160
    ctx.close()
