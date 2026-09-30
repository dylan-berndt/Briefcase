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
    expect(page.get_by_role("status").first).to_contain_text("Searching for: zebra-stripe")
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


def test_unrecognised_query(page):
    page.goto("/")
    search(page, "qwertyuiop")
    expect(page.get_by_text("No tags recognised")).to_contain_text("Not recognised: qwertyuiop")
    expect(page.locator(".ResultWindow")).to_have_count(0)


def test_feedback_needs_login_then_persists(page):
    page.goto("/")
    search(page, "serif")
    card = page.locator(".ResultWindow").first
    card.get_by_role("button", name="This font matches my search").click()
    expect(card.get_by_text("Log in to give feedback")).to_be_visible()
    expect(page.get_by_label("Username:")).to_be_visible()

    name = signUp(page)
    # logging in reloads the results with this user's state
    card = page.locator(".ResultWindow").first
    title = card.locator(".ResultTitle a").inner_text()
    yes = card.get_by_role("button", name="This font matches my search")
    yes.click()
    expect(yes).to_have_attribute("aria-pressed", "true")
    card.get_by_role("button", name="Rate 4 stars").click()
    expect(card.get_by_text("4 (1)")).to_be_visible()
    shot(page, "feedback")

    page.reload()
    card = page.locator(".ResultWindow").first
    expect(page.get_by_role("button", name=name)).to_be_visible()  # session survived the reload
    assert card.locator(".ResultTitle a").inner_text() == title
    expect(card.get_by_role("button", name="This font matches my search")).to_have_attribute("aria-pressed", "true")
    expect(card.get_by_text("4 (1)")).to_be_visible()

    # the vote belongs to that query; the rating to the font
    search(page, "serif bold")
    other = page.locator(".ResultWindow", has_text=title)
    if other.count():
        expect(other.get_by_role("button", name="This font matches my search")).to_have_attribute("aria-pressed", "false")
        expect(other.get_by_text("4 (1)")).to_be_visible()

    card = page.locator(".ResultWindow").first
    page.get_by_role("button", name=name).click()
    page.get_by_role("button", name="Log out").click()
    expect(page.get_by_role("button", name="Login")).to_be_visible()
    expect(page.locator(".ResultWindow").first.get_by_role("button", name="Rate 4 stars")).to_have_attribute("aria-pressed", "false")


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
