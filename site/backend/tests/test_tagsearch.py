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


def test_inflected_words_reach_their_base_tag(index):
    # "bolder" -> bold, "scripts" -> script, "not bolder" negates bold; a word with no known base stays unmatched
    assert index.parse("bolder scripts") == ([("bold", 1.0), ("script", 1.0)], [])
    assert index.parse("not bolder") == ([("bold", -1.0)], [])
    assert index.parse("glorping") == ([], ["glorping"])


def test_base_forms():
    from tagsearch import loadVocabularyClass
    baseForms = loadVocabularyClass().parse.__globals__["baseForms"]   # the parser module is loaded by path
    assert "drip" in baseForms("dripping") and "drip" in baseForms("drippy")
    assert "grunge" in baseForms("grungy") and "bubble" in baseForms("bubbly")
    assert "thin" in baseForms("thinner") and "curve" in baseForms("curves")


def inferableWord(index):
    """A word the caption table maps to tags this bundle has, which nothing else in the parser matches."""
    for word, groups in sorted(index.vocabulary.wordTags.items()):
        if any(g in index.groups for g in groups) and index.parse(word) == ([], [word]):
            return word
    pytest.skip("no caption-table word maps onto the fake bundle's tags")


def test_caption_table_only_used_on_request(index):
    word = inferableWord(index)
    # parse()/search() are unchanged: the word stays unmatched
    assert index.parse(word) == ([], [word])
    terms, unmatched, inferred = index.parseDetailed(word)
    assert terms == [] and unmatched == []
    assert len(inferred) == 1 and inferred[0][0] == word and inferred[0][2] == 0.5
    assert all(g in index.groups for g in inferred[0][1])


def test_inferred_words_can_be_ignored_and_negated(index):
    word = inferableWord(index)
    assert index.parseDetailed(word, ignore={word}) == ([], [word], [])
    _, _, inferred = index.parseDetailed("not " + word)
    assert inferred[0][2] == -0.5
    order, _, _, inferred, _ = index.searchDetailed(word)
    assert len(order) == index.numFonts and inferred



@pytest.fixture(scope="module")
def withSynonyms(fake):
    pytest.importorskip("en_core_web_md")
    index = TagIndex(Bundle(fake[0]), synonymModel="en_core_web_md")
    assert index.suggester is not None
    return index


def test_unknown_words_get_suggestions_not_scores(withSynonyms):
    # "ghastly" is not a phrase or a caption word; its neighbours (horror words) are suggested, not searched
    order, terms, unmatched, inferred, suggested = withSynonyms.searchDetailed("ghastly")
    assert terms == [] and inferred == [] and unmatched == ["ghastly"] and len(order) == 0
    assert suggested and suggested[0][0] == "ghastly"
    for group, via, similarity in suggested[0][1]:
        assert group in withSynonyms.groups and similarity >= withSynonyms.suggester.minSimilarity


def test_suggestions_skip_tags_already_in_the_query(withSynonyms):
    _, terms, _, _, suggested = withSynonyms.searchDetailed("horror ghastly")
    assert ("horror", 1.0) in terms
    assert all(group != "horror" for _, options in suggested for group, _, _ in options)


def test_added_tags_join_the_query(withSynonyms):
    order, terms, unmatched, _, _ = withSynonyms.searchDetailed("ghastly", tags=[("horror", 1.0), ("no-such-tag", 1.0)])
    assert ("horror", 1.0) in terms and "no-such-tag" in unmatched and len(order) == withSynonyms.numFonts


def test_suggestions_off_without_a_model(fake):
    index = TagIndex(Bundle(fake[0]), synonymModel="")
    assert index.suggester is None and index.searchDetailed("ghastly")[4] == []
