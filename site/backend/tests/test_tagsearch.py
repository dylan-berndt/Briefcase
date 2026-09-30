import numpy as np
import pytest

from bundle import Bundle
from bundleWriter import writeBundle
from tagsearch import TagIndex


@pytest.fixture(scope="module")
def index(fake):
    return TagIndex(Bundle(fake[0]))


def withGroup(planted, name):
    return {key for key, groups in planted.items() if name in groups}


def keysOf(index, order):
    return [index.bundle.fonts[i]["key"] for i in order]


def test_single_tag_puts_planted_fonts_first(index, planted):
    for name in ("serif", "bold", "script", "horror"):
        expected = withGroup(planted, name)
        assert expected
        order, terms, unmatched = index.search(name)
        assert terms == [(name, 1.0)] and unmatched == []
        assert set(keysOf(index, order[:len(expected)])) == expected


def test_alias_reaches_canonical_tag(index, planted):
    order, terms, _ = index.search("sans")
    assert terms == [("sans-serif", 1.0)]
    expected = withGroup(planted, "sans-serif")
    assert set(keysOf(index, order[:len(expected)])) == expected


def test_multi_tag_prefers_fonts_with_both(index, planted):
    groups = sorted({g for gs in planted.values() for g in gs} - {"zebra-stripe"})
    both = None
    for a in groups:
        for b in groups:
            if a < b and index.parse(f"{a} {b}")[0] == [(a, 1.0), (b, 1.0)]:
                common = withGroup(planted, a) & withGroup(planted, b)
                if common:
                    both = (a, b, common)
                    break
        if both:
            break
    assert both, "seed should plant at least one font with two queryable groups"
    a, b, common = both
    order, _, _ = index.search(f"{a} {b}")
    assert set(keysOf(index, order[:len(common)])) == common


def test_negation_sinks_fonts_with_the_tag(index, planted):
    expected = withGroup(planted, "serif")
    order, terms, _ = index.search("not serif")
    assert terms == [("serif", -1.0)]
    assert set(keysOf(index, order[-len(expected):])) == expected


def test_positive_and_negative_together(index, planted):
    bold, serif = withGroup(planted, "bold"), withGroup(planted, "serif")
    only = bold - serif
    order, terms, _ = index.search("bold not serif")
    assert dict(terms) == {"bold": 1.0, "serif": -1.0}
    top = keysOf(index, order[:len(only)])
    assert set(top) == only


def test_extra_model_tags_are_searchable(index, planted):
    expected = withGroup(planted, "zebra-stripe")
    order, terms, _ = index.search("zebra stripe")
    assert terms == [("zebra-stripe", 1.0)]
    assert set(keysOf(index, order[:len(expected)])) == expected


def test_junk_tag_names_are_not_searchable(index):
    assert "%E3%81%82%E3%81%84" not in index.groups


def test_nothing_matched(index):
    order, terms, unmatched = index.search("qwertyuiop")
    assert len(order) == 0 and terms == [] and unmatched == ["qwertyuiop"]
    order, terms, unmatched = index.search("   ")
    assert len(order) == 0 and terms == [] and unmatched == []


def test_partly_matched(index):
    _, terms, unmatched = index.search("serif qwerty")
    assert terms == [("serif", 1.0)] and unmatched == ["qwerty"]


def test_ranking_is_a_deterministic_permutation(index):
    first, _, _ = index.search("serif bold")
    second, _, _ = index.search("serif bold")
    assert (first == second).all()
    assert sorted(first) == list(range(index.numFonts))


def test_score_matches_a_dense_reference(index):
    logits = np.asarray(index.bundle.logits).astype(np.float64)  # [tags, fonts]
    p = 1 / (1 + np.exp(-logits))
    mass = p.sum(axis=0)
    rows = lambda name: index.groups[name]  # noqa: E731
    terms = [("serif", 1.0), ("bold", -1.0), ("thin", 0.6)]
    expected = (np.log(p[rows("serif")].sum(0) / mass)
                + np.log(1 - p[rows("bold")]).sum(0)
                + 0.6 * np.log(p[rows("thin")].sum(0) / mass))
    assert np.allclose(index.score(terms), expected, atol=2e-2)


def test_generic_fonts_do_not_win(tmp_path):
    """The point of the semantic multinomial: a font that scores high on every tag must not beat one that is
    specifically the queried tag, which a raw probability would tie."""
    vocab = ["alpha", "beta", "gamma", "delta"]
    logits = np.array([
        [4.0, 4.0, 4.0, 4.0],      # generic: high on everything
        [4.0, -6.0, -6.0, -6.0],   # specifically alpha
        [-6.0, -6.0, -6.0, -6.0],  # nothing
    ])
    fonts = [{"key": f"t:{i}", "name": f"F{i}", "source": "google", "url": "https://x", "creator": None}
             for i in range(3)]
    writeBundle(str(tmp_path), fonts, vocab, logits, [b"x"] * 3)
    index = TagIndex(Bundle(str(tmp_path)))
    order, terms, _ = index.search("alpha")
    assert terms == [("alpha", 1.0)]
    assert list(order)[:2] == [1, 0]


def test_canonical_without_model_tags_counts_as_unmatched(tmp_path):
    fonts = [{"key": "t:0", "name": "F", "source": "google", "url": "https://x", "creator": None}]
    writeBundle(str(tmp_path), fonts, ["alpha"], np.zeros((1, 1)), [b"x"])
    index = TagIndex(Bundle(str(tmp_path)))
    order, terms, unmatched = index.search("serif")
    assert len(order) == 0 and terms == [] and unmatched == ["serif"]
