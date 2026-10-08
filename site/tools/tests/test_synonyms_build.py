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
