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
    search(page, "elegant script not thin")
    tags = page.get_by_label("Tags in your search")
    expect(tags.get_by_role("button", name="elegant: included")).to_be_visible()
    boxes = tags.get_by_role("button").all()
    assert len(boxes) == 3 and len({round(b.bounding_box()["y"]) for b in boxes}) == 1   # words side by side
    assert tags.get_by_role("listitem").first.bounding_box()["x"] < boxes[0].bounding_box()["x"]  # word, then its box
    assert page.get_by_text("Searching for:").count() == 0       # no label, just the words


def test_tag_line_ticks_are_page_state_and_change_the_results(page):
    page.goto("/")
    requests = []
    page.on("request", lambda r: requests.append(r.url) if "/api/font/" in r.url else None)
    search(page, "elegant script not thin airy slimy")
    tags = page.get_by_label("Tags in your search")
    expect(tags.get_by_role("button", name="elegant: included")).to_be_visible()
    expect(tags.get_by_role("button", name="thin: excluded")).to_be_visible()
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
    # tick -> empty -> cross -> empty -> tick on a typed tag; every step changes the fonts shown
    page.get_by_role("button", name="script: included").click()
    expect(tags.get_by_role("button", name="script: off")).to_be_visible()
    settles(lambda r: r != before)
    off = results()
    tags.get_by_role("button", name="script: off").click()
    expect(tags.get_by_role("button", name="script: excluded")).to_be_visible()
    settles(lambda r: r != off)
    tags.get_by_role("button", name="script: excluded").click()
    expect(tags.get_by_role("button", name="script: off")).to_be_visible()
    settles(lambda r: r == off)
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
