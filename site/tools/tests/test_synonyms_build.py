import argparse
import json
import os

import pytest

nltk = pytest.importorskip("nltk")
try:
    from nltk.corpus import wordnet
    wordnet.ensure_loaded()
except LookupError:
    pytest.skip("the WordNet data is not installed (python -m nltk.downloader wordnet)", allow_module_level=True)

import buildSynonyms

TAGS = ["drip", "horror", "sloppy", "steam", "elegant", "thin", "script", "baroque", "dingbat", "grunge", "vintage"]


@pytest.fixture(scope="module")
def builder(tmp_path_factory):
    vocab = tmp_path_factory.mktemp("v") / "vocab.json"
    vocab.write_text(json.dumps(TAGS))
    args = argparse.Namespace(vocabulary=os.path.join(buildSynonyms.REPO, "configs", "tagVocabulary.json"), modelVocab=str(vocab), minScore=0.05, topK=8,
                              minSenseWeight=0.05)
    return buildSynonyms.Builder(args, wordnet)


def names(found):
    return [tag for tag, _, _ in found]


def test_wet_reaches_drip_through_drippy(builder):
    found = {tag: via for tag, _, via in builder.suggest("wet")}
    assert found.get("drip") == "drippy"


def test_verb_senses_do_not_count(builder):
    # "fancy" the verb is visualize/picture/image, which the caption table ties to dingbat; the adjective is not
    assert "dingbat" not in names(builder.suggest("fancy"))


def test_scores_are_probabilities_best_first(builder):
    found = builder.suggest("graceful")
    scores = [score for _, score, _ in found]
    assert found and all(0 < s <= 1 for s in scores) and scores == sorted(scores, reverse=True)


def test_a_word_the_search_already_understands_is_left_out_of_the_table(builder):
    assert builder.tagsOf("elegant")          # understood as typed, so main() skips it
    assert not builder.tagsOf("wet")


def test_unknown_words_have_no_suggestions(builder):
    assert builder.suggest("glorping") == []


# ---- with Datamuse (a fake: explicit answers, no network)

class FakeDatamuse:
    def __init__(self, data):
        self.data = data                                   # {(rel, word): [words]}

    def get(self, rel, word, n=30):
        found = self.data.get((rel, word))
        return None if found is None else found[:n]

    def prefetch(self, questions, label=""):
        return [q for q in questions if self.get(q[0], q[1]) is None]


@pytest.fixture
def combined(tmp_path):
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps(TAGS))
    args = argparse.Namespace(vocabulary=os.path.join(buildSynonyms.REPO, "configs", "tagVocabulary.json"), modelVocab=str(vocab), minScore=0.05, topK=8,
                              minSenseWeight=0.05)

    def make(data):
        return buildSynonyms.Builder(args, wordnet, FakeDatamuse(data))
    return make


def test_a_means_like_word_reaches_a_tag(combined):
    builder = combined({("pos", "squelchy"): ["adj"], ("rel_ant", "squelchy"): [], ("ml", "squelchy"): ["drippy", "xyzzy"]})
    assert {tag: via for tag, _, via in builder.suggest("squelchy")}.get("drip") == "drippy"


def test_near_the_top_of_the_means_like_list_the_rank_barely_matters(combined):
    fillers = [f"filler{i}" for i in range(10)]
    first = combined({("pos", "squelchy"): ["adj"], ("rel_ant", "squelchy"): [], ("ml", "squelchy"): ["drippy"] + fillers})
    eleventh = combined({("pos", "squelchy"): ["adj"], ("rel_ant", "squelchy"): [], ("ml", "squelchy"): fillers + ["drippy"]})
    top, lower = dict((t, s) for t, s, _ in first.suggest("squelchy"))["drip"], dict((t, s) for t, s, _ in eleventh.suggest("squelchy"))["drip"]
    assert lower >= 0.8 * top


def test_a_related_word_on_a_rare_sense_keeps_only_the_floor(combined):
    # WordNet ties "antiquated" to the main sense of "old" and "genuine" only to a rare one (0.005 of its use)
    builder = combined({("pos", "old"): ["adj"], ("rel_ant", "old"): [], ("ml", "old"): ["genuine", "antiquated"]})
    related, dropped = builder.candidates("old")
    assert dropped == {}
    assert related["genuine"] == pytest.approx((buildSynonyms.SOFT_FLOOR / (1 + 0 / buildSynonyms.ML_FLATNESS), "ml"))
    assert related["antiquated"][0] > related["genuine"][0]            # its sense is 44% of the word's use


def test_the_opposite_side_of_a_word_is_dropped(combined):
    builder = combined({
        ("pos", "zorbly"): ["adj"], ("rel_ant", "zorbly"): ["spotless"],
        ("ml", "zorbly"): ["drippy", "elegant", "vintage", "baroque"],
        ("ml", "spotless"): ["elegant", "immaculate"], ("rel_syn", "spotless"): ["pristine"],
        ("ml", "drippy"): ["moist"],
        ("ml", "vintage"): ["old", "spotless"],            # its own list contains the antonym
        ("ml", "baroque"): ["ornate"], ("ml", "elegant"): ["graceful"],
    })
    related, dropped = builder.candidates("zorbly")
    assert dropped == {"elegant": "opposite side", "vintage": "means like an antonym"}
    names = {tag for tag, _, _ in builder.suggest("zorbly")}
    assert {"drip", "baroque"} <= names and not names & {"elegant", "vintage"}


def test_a_noun_takes_the_adjectives_that_modify_it(combined):
    # no ml entry: if the builder asked for one it would find the data missing and give up (None)
    builder = combined({("pos", "zorb"): ["n"], ("rel_ant", "zorb"): [], ("rel_jjb", "zorb"): ["old", "xyzzy"]})
    assert "vintage" in {tag for tag, _, _ in builder.suggest("zorb")}


def test_missing_datamuse_data_means_no_answer_not_an_empty_one(combined):
    assert combined({}).suggest("squelchy") is None
    assert combined({("pos", "squelchy"): ["adj"], ("rel_ant", "squelchy"): []}).suggest("squelchy") is None


def test_prefetch_asks_for_everything_suggest_needs(combined):
    asked = []

    class Recording(FakeDatamuse):
        def prefetch(self, questions, label=""):
            asked.extend(questions)
            return super().prefetch(questions, label)
    builder = combined({})
    builder.datamuse = Recording({("pos", "squelchy"): ["adj"], ("rel_ant", "squelchy"): ["spotless"], ("ml", "squelchy"): ["drippy"]})
    builder.prefetch(["squelchy"])
    assert ("pos", "squelchy", 1) in asked and ("ml", "squelchy", 30) in asked and ("rel_ant", "squelchy", 12) in asked
    assert ("ml", "spotless", 30) in asked and ("rel_syn", "spotless", 10) in asked and ("ml", "drippy", 30) in asked


def test_related_synsets_come_in_a_fixed_order(builder):
    # nltk returns pointers in hash-seed order; the build sorts them so a rebuild gives the same table
    for sense in wordnet.synsets("wet") + wordnet.synsets("rough"):
        pairs = builder.related(sense)
        assert [r for r, _ in pairs][:1] == ["syn"]
        for relation in ("similar", "also"):
            names = [o.name() for r, o in pairs if r == relation]
            assert names == sorted(names)


def test_a_tie_between_related_words_goes_to_the_alphabetically_first(combined):
    builder = combined({("pos", "squelchy"): ["adj"], ("rel_ant", "squelchy"): [], ("ml", "squelchy"): ["dripping", "drippy"]})
    # "dripping" is the tag as typed (x1.0), "drippy" a caption-table guess (x0.5): these two contribute exactly 0.25 each
    related = {"drippy": (0.5, "ml"), "dripping": (0.25, "ml")}
    assert builder.tagsOf("dripping")["drip"] == 1.0 and builder.tagsOf("drippy")["drip"] == 0.5
    assert {t: v for t, s, v in builder.score(related)}["drip"] == "dripping"
