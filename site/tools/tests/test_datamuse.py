import json
import urllib.error

import pytest

import datamuse


class Reply:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


class Opener:
    """Stands in for urlopen: hands out the answers in turn (the last one repeats); an Exception is raised instead."""
    def __init__(self, *answers):
        self.answers, self.urls = list(answers), []

    def __call__(self, request, timeout=None):
        self.urls.append(request.full_url)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return Reply(answer)


def make(tmp_path, opener, **kw):
    return datamuse.Datamuse(str(tmp_path / "cache.sqlite"), opener=opener, sleep=lambda seconds: None, perSecond=1e6, **kw)


def http(code):
    return urllib.error.HTTPError("https://api.datamuse.com/words", code, "x", {}, None)


MEANS_LIKE = [{"word": "moist", "score": 9}, {"word": "damp", "score": 8}]


def test_a_question_is_asked_once_and_remembered(tmp_path):
    opener = Opener(MEANS_LIKE)
    client = make(tmp_path, opener)
    assert client.get("ml", "wet", 30) == ["moist", "damp"]
    assert client.get("ml", "wet", 30) == ["moist", "damp"]
    assert len(opener.urls) == 1 and "ml=wet" in opener.urls[0] and "max=30" in opener.urls[0]


def test_the_cache_survives_a_restart_and_serves_an_offline_run(tmp_path):
    with make(tmp_path, Opener(MEANS_LIKE)) as first:
        first.get("ml", "wet", 30)
    opener = Opener(MEANS_LIKE)
    offline = make(tmp_path, opener, offline=True)
    assert offline.get("ml", "wet", 30) == ["moist", "damp"]
    assert offline.get("ml", "dry", 30) is None            # never asked, and offline will not ask
    assert opener.urls == []


def test_part_of_speech_checks_the_answer_is_the_word_itself(tmp_path):
    exact = [{"word": "wet", "tags": ["adj", "n", "f:3.1", "prop"]}]
    assert make(tmp_path, Opener(exact)).get("pos", "wet") == ["adj", "n"]
    other = [{"word": "wit", "tags": ["n"]}]               # sp= is a spelling pattern: the first hit can be another word
    assert make(tmp_path / "b", Opener(other)).get("pos", "wet") == []


def test_a_busy_server_is_retried(tmp_path):
    opener = Opener(http(503), http(429), MEANS_LIKE)
    assert make(tmp_path, opener).get("ml", "wet", 30) == ["moist", "damp"]
    assert len(opener.urls) == 3


def test_a_failed_request_raises_and_is_not_cached(tmp_path):
    opener = Opener(http(404))
    client = make(tmp_path, opener)
    with pytest.raises(datamuse.DatamuseError):
        client.get("ml", "wet", 30)
    assert len(opener.urls) == 1                           # a 404 is not worth retrying
    assert client.cached("ml", "wet", 30) is None

    giving_up = Opener(http(500))
    flaky = make(tmp_path / "b", giving_up, retries=3)
    with pytest.raises(datamuse.DatamuseError):
        flaky.get("ml", "wet", 30)
    assert len(giving_up.urls) == 3 and flaky.cached("ml", "wet", 30) is None


def test_an_answer_that_is_not_a_list_is_an_error(tmp_path):
    with pytest.raises(datamuse.DatamuseError):
        make(tmp_path, Opener({"error": "nope"})).get("ml", "wet", 30)


def test_prefetch_fills_the_cache_and_reports_what_failed(tmp_path):
    def opener(request, timeout=None):
        if "dry" in request.full_url:
            raise http(404)
        return Reply(MEANS_LIKE)
    client = make(tmp_path, opener)
    failed = client.prefetch([("ml", "wet", 30), ("ml", "dry", 30), ("ml", "wet", 30)])
    assert failed == [("ml", "dry", 30)]
    assert client.cached("ml", "wet", 30) == ["moist", "damp"]
    assert client.prefetch([("ml", "wet", 30)]) == []      # already there
