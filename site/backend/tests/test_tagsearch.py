import json

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


def plantedWords(index, planted, count=4):
    """Groups that were planted in some font and are searched by their own one-word name (the vocabulary is large, so
    the fake bundle plants only some of it)."""
    names = sorted({g for gs in planted.values() for g in gs})
    words = [g for g in names if g.isalpha() and index.parse(g)[0] == [(g, 1.0)]]
    assert len(words) >= count
    return words[:count]


def test_single_tag_puts_planted_fonts_first(index, planted):
    for name in plantedWords(index, planted):
        expected = withGroup(planted, name)
        assert expected
        order, terms, unmatched = index.search(name)
        assert terms == [(name, 1.0)] and unmatched == []
        assert set(keysOf(index, order[:len(expected)])) == expected


def test_alias_reaches_canonical_tag(index, planted):
    # a phrase that is not itself a tag reaches the group it was written for
    for name in sorted({g for gs in planted.values() for g in gs}):
        for phrase in sorted(index.vocabulary.canonical[name]["aliases"]):
            if " " in phrase and index.parse(phrase)[0] == [(name, 1.0)]:
                expected = withGroup(planted, name)
                order, terms, _ = index.search(phrase)
                assert terms == [(name, 1.0)]
                assert set(keysOf(index, order[:len(expected)])) == expected
                return
    pytest.skip("no planted group has a multi-word alias")


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


def test_negation_is_not_searched(index, planted):
    # "not serif" means nothing: no tag for serif, and "bold not serif" ranks exactly like "bold"
    assert index.search("not serif")[1] == []
    assert len(index.search("not serif")[0]) == 0
    order, terms, _ = index.search("bold not serif")
    assert terms == [("bold", 1.0)]
    assert (order == index.search("bold")[0]).all()


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
    terms = [("serif", 1.0), ("bold", 0.3), ("thin", 0.6)]
    expected = (np.log(p[rows("serif")].sum(0) / mass)
                + 0.3 * np.log(p[rows("bold")].sum(0) / mass)
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


def test_inflected_words_reach_their_tag_by_stem(index):
    # stems are compared on both sides: "scripts" -> script, "swirling" -> the phrase "swirls"; a negated
    # one is dropped by describe(); a word whose stem no phrase shares stays unmatched
    vocabulary = index.vocabulary
    assert vocabulary.stem is not None
    assert index.parse("scripts") == ([("script", 1.0)], [])
    assert index.describe("not scripts")[0] == []
    assert index.parse("swirling")[0] == [("swirl", 1.0)]
    assert index.parse("glorping") == ([], ["glorping"])


def test_stems_do_not_chop_adjectives():
    # the suffix rules this replaced turned "slimy" into "slim"; a stemmer applied to both sides cannot
    from tagsearch import findVocabularyConfig, loadVocabularyClass
    vocabulary = loadVocabularyClass()(findVocabularyConfig())
    assert vocabulary.stemMatch("slimy", vocabulary.aliasStems) is None
    assert vocabulary.stemMatch("sketched", vocabulary.aliasStems) == "sketch"
    assert vocabulary.stemMatch("classically", vocabulary.aliasStems) == "classical"


@pytest.mark.parametrize("query", ["1940", "1940's", "1940s", "40", "40s", "'40s", "40's"])
def test_decades_are_found_however_they_are_written(query):
    from tagsearch import findVocabularyConfig, loadVocabularyClass
    vocabulary = loadVocabularyClass()(findVocabularyConfig())
    weights, unmatched = vocabulary.parse(query)
    assert list(weights) == ["1940s"] and unmatched == []


def test_years_outside_a_decade_tag_stay_unmatched():
    from tagsearch import findVocabularyConfig, loadVocabularyClass
    vocabulary = loadVocabularyClass()(findVocabularyConfig())
    assert vocabulary.parse("2008") == ({}, ["2008"])


def inferableWord(index):
    """A word the caption table maps to tags this bundle has, which nothing else in the parser matches."""
    for word, groups in sorted(index.vocabulary.wordTags.items()):
        if any(g in index.groups for g in groups) and index.parse(word) == ([], [word]):
            return word
    pytest.skip("no caption-table word maps onto the fake bundle's tags")


def test_caption_table_is_separate_from_parse(index):
    word = inferableWord(index)
    # parse() is unchanged: the word stays unmatched; parseDetailed() reports the guess
    assert index.parse(word) == ([], [word])
    terms, unmatched, inferred = index.parseDetailed(word)
    assert terms == [] and unmatched == []
    assert len(inferred) == 1 and inferred[0][0] == word and inferred[0][2] == 0.5
    assert all(g in index.groups for g in inferred[0][1])


def test_guessed_words_become_ordinary_tags_at_half_weight(index):
    word = inferableWord(index)
    inferred = index.parseDetailed(word)[2]
    terms, suggested, left = index.describe(word)
    assert left == [] and suggested == []
    assert {g for g, _ in terms} == set(inferred[0][1]) and all(w == 0.5 for _, w in terms)
    assert index.describe("not " + word)[0] == []
    order, _, _ = index.search(word)
    assert len(order) == index.numFonts


def test_a_guessed_tag_is_not_listed_twice(index):
    word = inferableWord(index)
    guessed = index.describe(word)[0]
    both = index.describe(f"{guessed[0][0]} {word}")[0]
    assert [n for n, _ in both].count(guessed[0][0]) == 1
    assert dict(both)[guessed[0][0]] == 1.0          # what was typed wins over the guess


def test_rank_orders_exactly_the_given_tags(index, planted):
    expected = withGroup(planted, "serif")
    order = index.rank([("serif", 1.0)])
    assert set(keysOf(index, order[:len(expected)])) == expected
    assert len(index.rank([])) == 0
    assert (index.rank([("serif", 1.0), ("bold", 1.0)]) == index.search("serif bold")[0]).all()


def test_choices_parse_weights(index):
    assert index.parseChoices("serif,bold:0.6, Script:0.5 ") == [("serif", 1.0), ("bold", 0.6), ("script", 0.5)]
    assert index.parseChoices("") == []
    assert index.parseChoices("serif,serif:0.5") == [("serif", 0.5)]          # the last one wins
    assert index.parseChoices("no-such-tag,serif") == [("serif", 1.0)]         # unknown names are skipped


@pytest.mark.parametrize("text", ["-serif", "serif:abc", "serif:0", "serif:1.5", "serif:-0.2", "serif:nan"])
def test_bad_choices_are_rejected(index, text):
    with pytest.raises(ValueError):
        index.parseChoices(text)


def test_too_many_choices_are_rejected(index):
    with pytest.raises(ValueError, match="at most"):
        index.parseChoices(",".join(["serif"] * 65))


TABLE = {"source": "test", "settings": {}, "words": {
    "ghastly": [["horror", 0.52, "grisly"], ["no-such-tag", 0.5, "x"], ["grunge", 0.31, "gruesome"], ["bold", 0.04, "y"]],
    "drench": [["elegant", 0.4, "soak"], ["script", 0.3, "wet"]],
    "many": [[t, 0.9 - i / 100, "w"] for i, t in enumerate(["bold", "thin", "serif", "script", "horror", "grunge",
                                                         "elegant", "vintage", "no-such-tag"])],
}}


@pytest.fixture(scope="module")
def withSynonyms(fake, tmp_path_factory):
    path = tmp_path_factory.mktemp("syn") / "synonymTags.json"
    path.write_text(json.dumps(TABLE))
    index = TagIndex(Bundle(fake[0]), synonymTable=str(path))
    assert index.suggester is not None
    return index


def test_unknown_words_get_suggestions_not_scores(withSynonyms):
    terms, suggested, left = withSynonyms.describe("ghastly")
    assert terms == [] and left == []
    # best first, only tags the model can score, and nothing under the table's own score floor
    assert suggested == [("horror", "grisly", 0.52), ("grunge", "gruesome", 0.31)]
    assert len(withSynonyms.search("ghastly")[0]) == 0           # suggestions are not part of the search


def test_suggestions_are_capped_and_ordered(withSynonyms):
    _, suggested, _ = withSynonyms.describe("many")
    assert [g for g, _, _ in suggested] == ["bold", "thin", "serif", "script", "horror", "grunge", "elegant", "vintage"]


def test_suggestions_skip_tags_already_in_the_query(withSynonyms):
    terms, suggested, _ = withSynonyms.describe("horror ghastly")
    assert ("horror", 1.0) in terms
    assert [g for g, _, _ in suggested] == ["grunge"]


def test_inflected_unknown_words_find_their_table_word_by_stem(withSynonyms):
    assert [g for g, _, _ in withSynonyms.describe("drenched")[1]] == ["elegant", "script"]


def test_a_word_with_no_entry_is_unmatched(withSynonyms):
    assert withSynonyms.describe("glorping") == ([], [], ["glorping"])


def test_suggestions_off_without_a_table(fake):
    index = TagIndex(Bundle(fake[0]), synonymTable="")
    assert index.suggester is None and index.describe("ghastly") == ([], [], ["ghastly"])


def test_the_shipped_table_is_well_formed():
    from synonyms import findSynonymTable
    path = findSynonymTable()
    assert path, "configs/synonymTags.json is missing"
    with open(path, encoding="utf-8") as f:
        words = json.load(f)["words"]
    assert len(words) > 1000
    for word, entries in list(words.items())[:500]:
        assert word.isalpha() and entries
        for tag, score, via in entries:
            assert isinstance(tag, str) and 0 < score <= 1 and isinstance(via, str)
        assert [e[1] for e in entries] == sorted((e[1] for e in entries), reverse=True)


def splitBundle(tmp_path):
    """40 fonts that all have technical. wide splits them in half, narrow is its mirror, rounded nearly duplicates
    wide, serif splits them independently (odd/even), and script is on none of them."""
    vocab = ["technical", "wide", "narrow", "rounded", "serif", "script"]
    logits = np.full((40, len(vocab)), -6.0)
    logits[:, 0] = 6.0
    logits[:20, 1] = 6.0
    logits[20:, 2] = 6.0
    logits[:19, 3] = 6.0
    logits[::2, 4] = 6.0
    fonts = [{"key": f"t:{i}", "name": f"F{i}", "source": "google", "url": "https://x", "creator": None}
             for i in range(40)]
    writeBundle(str(tmp_path), fonts, vocab, logits, [b"x"] * 40)
    return TagIndex(Bundle(str(tmp_path)))


def test_refinements_split_the_top_fonts_without_repeating_a_split(tmp_path):
    index = splitBundle(tmp_path)
    refinements = index.refinements([("technical", 1.0)], top=40)
    offered = {name: (share, opposites) for name, share, opposites in refinements}
    # one of wide/narrow is offered with the other as its opposite; rounded splits like wide so it is dropped;
    # serif is a different split; technical is already in the query and script splits nothing
    assert set(offered) == {"wide", "serif"} or set(offered) == {"narrow", "serif"}
    first = "wide" if "wide" in offered else "narrow"
    other = "narrow" if first == "wide" else "wide"
    assert [name for name, _ in offered[first][1]] == [other]
    assert offered["serif"][1] == []
    for share, opposites in offered.values():
        assert abs(share - 0.5) < 0.01
        assert all(abs(s - 0.5) < 0.01 for _, s in opposites)


def test_refinements_respect_count_and_need_a_query(tmp_path):
    index = splitBundle(tmp_path)
    assert len(index.refinements([("technical", 1.0)], top=40, count=1)) == 1
    assert index.refinements([]) == []


def test_refinements_on_the_fake_bundle(index, planted):
    name = plantedWords(index, planted, count=1)[0]
    refinements = index.refinements([(name, 1.0)])
    assert refinements and len(refinements) <= 8
    assert name not in {n for n, _, _ in refinements}
    offered = [n for n, _, _ in refinements]
    assert len(set(offered)) == len(offered)
