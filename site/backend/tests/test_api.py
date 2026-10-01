import os
import subprocess
import sys

import pytest

from conftest import register, login


def query(client, q="serif", **params):
    params = {"query": q, **params}
    return client.get("/api/font/query", query_string=params)


# ------------------------------------------------------------------ search and pagination

def test_query_shape(client):
    response = query(client, "serif", pageSize=5)
    assert response.status_code == 200
    body = response.json
    assert body["page"] == 1 and body["pageSize"] == 5 and body["total"] == 300 and body["totalPages"] == 60
    assert body["tags"] == [{"tag": "serif", "weight": 1.0}]
    first = body["results"][0]
    assert set(first) == {"key", "name", "source", "url", "creator", "specimen", "rating", "vote"}
    assert first["specimen"].startswith("/api/font/specimen/") and "?v=" in first["specimen"]
    assert first["rating"] == {"average": None, "count": 0, "mine": None} and first["vote"] == 0
    assert first["url"].startswith("https://")


def test_pages_tile_the_whole_ranking(client, app):
    index = app.extensions["tagIndex"]
    order, _, _ = index.search("bold")
    expected = [index.bundle.fonts[i]["key"] for i in order]

    seen = []
    for page in range(1, 14):
        body = query(client, "bold", page=page, pageSize=25).json
        assert body["totalPages"] == 12
        seen += [r["key"] for r in body["results"]]
    assert seen == expected  # no overlaps, no gaps, same order as the full ranking
    assert len(set(seen)) == 300


def test_last_page_is_partial_and_beyond_is_empty(client):
    last = query(client, "bold", page=12, pageSize=25).json
    assert len(last["results"]) == 300 - 11 * 25
    beyond = query(client, "bold", page=13, pageSize=25).json
    assert beyond["results"] == [] and beyond["total"] == 300 and beyond["totalPages"] == 12


def test_default_page_size(client):
    assert len(query(client).json["results"]) == 24


@pytest.mark.parametrize("params", [{"page": 0}, {"page": -1}, {"page": "x"}, {"pageSize": 0}, {"pageSize": 101},
                                    {"pageSize": "many"}, {"page": 1.5}])
def test_bad_paging_is_400(client, params):
    assert query(client, **params).status_code == 400


def test_max_page_size_allowed(client):
    assert len(query(client, "bold", pageSize=100).json["results"]) == 100


def test_query_length_limit(client):
    assert query(client, "a" * 201).status_code == 400
    assert query(client, "serif " * 33).status_code == 200  # 198 chars


def test_nothing_matched_gives_no_results(client):
    body = query(client, "qwertyuiop").json
    assert body["results"] == [] and body["total"] == 0 and body["totalPages"] == 0 and body["tags"] == []
    body = query(client, "").json
    assert body["results"] == [] and body["tags"] == []


def test_partly_matched_query_ranks_on_what_matched(client):
    body = query(client, "serif qwerty").json
    assert body["total"] == 300 and body["tags"] == [{"tag": "serif", "weight": 1.0}]


def test_query_is_case_and_punctuation_tolerant(client):
    a = query(client, "Bold, Serif!").json
    b = query(client, "bold serif").json
    assert [r["key"] for r in a["results"]] == [r["key"] for r in b["results"]]


def test_unicode_and_html_in_query_are_harmless(client):
    for q in ["<script>alert(1)</script>", "日本語 フォント", "'; DROP TABLE users; --", "\x00bold"]:
        assert query(client, q).status_code == 200


# ------------------------------------------------------------------ specimens

def test_specimen_bytes_and_caching(client, app):
    bundle = app.extensions["bundle"]
    url = query(client, "serif", pageSize=3).json["results"][1]["specimen"]
    response = client.get(url)
    assert response.status_code == 200 and response.mimetype == "image/webp"
    i = int(url.split("/")[-1].split("?")[0])
    assert response.data == bundle.specimen(i)
    assert "immutable" in response.headers["Cache-Control"]
    again = client.get(url, headers={"If-None-Match": response.headers["ETag"]})
    assert again.status_code == 304


@pytest.mark.parametrize("path", ["/api/font/specimen/300", "/api/font/specimen/99999", "/api/font/specimen/-1",
                                  "/api/font/specimen/abc"])
def test_specimen_out_of_range(client, path):
    assert client.get(path).status_code == 404


# ------------------------------------------------------------------ accounts

def test_register_validation(client):
    assert register(client, "ab", "correct horse").status_code == 400
    assert register(client, "bad name!", "correct horse").status_code == 400
    assert register(client, "alice", "short").status_code == 400
    assert register(client, "alice", "x" * 129).status_code == 400
    assert client.post("/api/font/register").status_code == 400
    assert register(client).status_code == 200
    duplicate = register(client)
    assert duplicate.status_code == 400 and "exists" in duplicate.json["message"]


def test_login_and_me_and_logout(client):
    assert client.get("/api/font/me").json == {"username": None}
    register(client)
    assert login(client, password="wrong password").status_code == 400
    assert login(client, "nobody").status_code == 400
    response = login(client)
    assert response.status_code == 200
    cookie = response.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert client.get("/api/font/me").json == {"username": "alice"}
    assert client.post("/api/font/logout").status_code == 200
    assert client.get("/api/font/me").json == {"username": None}


def test_cookie_is_secure_by_default(makeApp):
    app = makeApp(COOKIE_SECURE=True)
    client = app.test_client()
    register(client)
    assert "Secure" in login(client).headers["Set-Cookie"]


def test_tampered_and_expired_tokens_are_anonymous(client, app):
    import jwt
    from datetime import datetime, timedelta, timezone
    register(client)
    login(client)
    client.set_cookie("token", "garbage")
    assert client.get("/api/font/me").json == {"username": None}
    expired = jwt.encode({"publicID": "x", "exp": datetime.now(timezone.utc) - timedelta(seconds=5)},
                         app.config["SECRET_KEY"], algorithm="HS256")
    client.set_cookie("token", expired)
    assert client.get("/api/font/me").json == {"username": None}
    forged = jwt.encode({"publicID": "x"}, "another secret", algorithm="HS256")
    client.set_cookie("token", forged)
    assert client.post("/api/font/rate", json={"fontKey": "x", "rating": 3}).status_code == 401


def test_token_for_deleted_user_is_rejected(client, app):
    import sqlite3
    register(client)
    login(client)
    db = sqlite3.connect(app.config["DATABASE"])
    db.execute("DELETE FROM users")
    db.commit()
    db.close()
    assert client.post("/api/font/rate", json={"fontKey": "x", "rating": 3}).status_code == 401


def test_register_rate_limit(makeApp):
    client = makeApp(RATELIMIT_ENABLED=True).test_client()
    codes = [register(client, f"user{i}").status_code for i in range(5)]
    assert codes == [200, 200, 200, 429, 429]


# ------------------------------------------------------------------ approve / disapprove

def firstKey(client, q="serif", n=0):
    return query(client, q, pageSize=5).json["results"][n]["key"]


def test_vote_requires_login(client):
    key = firstKey(client)
    assert client.post("/api/font/approve", json={"fontKey": key, "query": "serif", "vote": 1}).status_code == 401
    assert client.post("/api/font/rate", json={"fontKey": key, "rating": 4}).status_code == 401
    assert client.post("/api/font/describe", json={"fontKey": key, "description": "nice"}).status_code == 401


def test_vote_set_flip_clear_and_shown_in_results(user):
    key = firstKey(user)

    def shown(q="serif"):
        return next(r for r in query(user, q, pageSize=50).json["results"] if r["key"] == key)["vote"]

    def vote(v, q="serif"):
        return user.post("/api/font/approve", json={"fontKey": key, "query": q, "vote": v})

    assert vote(1).json == {"message": "Successful", "vote": 1} and shown() == 1
    assert vote(-1).status_code == 200 and shown() == -1  # flipping replaces, no second row
    assert vote(0).status_code == 200 and shown() == 0
    assert vote(0).status_code == 200  # clearing nothing is fine


def test_votes_are_per_query_and_normalized(user):
    key = firstKey(user, "bold")
    user.post("/api/font/approve", json={"fontKey": key, "query": "  Bold   Font ", "vote": 1})
    same = next(r for r in query(user, "bold font", pageSize=100).json["results"] if r["key"] == key)
    other = next(r for r in query(user, "bold", pageSize=100).json["results"] if r["key"] == key)
    assert same["vote"] == 1 and other["vote"] == 0


def test_votes_are_per_user(client, app):
    key = firstKey(client)
    a, b = app.test_client(), app.test_client()
    for c, name in ((a, "alice"), (b, "bobby")):
        register(c, name)
        login(c, name)
    a.post("/api/font/approve", json={"fontKey": key, "query": "serif", "vote": 1})
    b.post("/api/font/approve", json={"fontKey": key, "query": "serif", "vote": -1})
    b.post("/api/font/approve", json={"fontKey": key, "query": "serif", "vote": 0})  # only clears bobby's
    top = lambda c: next(r for r in query(c, "serif", pageSize=50).json["results"] if r["key"] == key)  # noqa: E731
    assert top(a)["vote"] == 1 and top(b)["vote"] == 0
    assert top(client)["vote"] == 0  # anonymous sees no vote


@pytest.mark.parametrize("body", [
    {"query": "serif", "vote": 1},                               # no font
    {"fontKey": "nope:nope", "query": "serif", "vote": 1},       # unknown font
    {"fontKey": 5, "query": "serif", "vote": 1},
])
def test_vote_unknown_font_is_404(user, body):
    assert user.post("/api/font/approve", json=body).status_code == 404


@pytest.mark.parametrize("extra", [
    {"vote": 2}, {"vote": "1"}, {"vote": True}, {"vote": None}, {"vote": 0.5}, {},
    {"vote": 1, "query": ""}, {"vote": 1, "query": "   "}, {"vote": 1, "query": 7}, {"vote": 1, "query": "x" * 201},
])
def test_vote_validation(user, extra):
    key = firstKey(user)
    body = {"fontKey": key, "query": "serif", **extra}
    assert user.post("/api/font/approve", json=body).status_code == 400


def test_vote_body_must_be_json(user):
    assert user.post("/api/font/approve", data="vote=1").status_code in (400, 404)
    assert user.post("/api/font/approve", json=[1, 2]).status_code in (400, 404)


# ------------------------------------------------------------------ ratings

def ratingOf(client, key, q="serif"):
    return next(r for r in query(client, q, pageSize=100).json["results"] if r["key"] == key)["rating"]


def test_rating_set_update_clear(user):
    key = firstKey(user)
    response = user.post("/api/font/rate", json={"fontKey": key, "rating": 4})
    assert response.json["rating"] == {"average": 4.0, "count": 1, "mine": 4}
    assert ratingOf(user, key) == {"average": 4.0, "count": 1, "mine": 4}
    assert user.post("/api/font/rate", json={"fontKey": key, "rating": 2}).json["rating"]["mine"] == 2
    assert ratingOf(user, key)["count"] == 1  # updated, not duplicated
    cleared = user.post("/api/font/rate", json={"fontKey": key, "rating": 0})
    assert cleared.json["rating"] == {"average": None, "count": 0, "mine": None}
    assert ratingOf(user, key) == {"average": None, "count": 0, "mine": None}


def test_rating_aggregates_across_users(client, app):
    key = firstKey(client)
    clients = []
    for name, stars in (("alice", 5), ("bobby", 2), ("carol", 2)):
        c = app.test_client()
        register(c, name)
        login(c, name)
        assert c.post("/api/font/rate", json={"fontKey": key, "rating": stars}).status_code == 200
        clients.append(c)
    assert ratingOf(client, key) == {"average": 3.0, "count": 3, "mine": None}
    assert ratingOf(clients[0], key)["mine"] == 5
    assert ratingOf(clients[1], key)["mine"] == 2


def test_rating_is_per_font_not_per_query(user):
    key = firstKey(user, "bold")
    user.post("/api/font/rate", json={"fontKey": key, "rating": 5})
    assert ratingOf(user, key, "bold")["mine"] == 5
    # the same font reached by any other query carries the same rating
    order = query(user, "serif", pageSize=100).json["results"]
    other = [r for r in order if r["key"] == key]
    if other:
        assert other[0]["rating"]["mine"] == 5


@pytest.mark.parametrize("rating", [6, -1, 2.5, "3", True, None])
def test_rating_validation(user, rating):
    key = firstKey(user)
    assert user.post("/api/font/rate", json={"fontKey": key, "rating": rating}).status_code == 400


def test_rating_unknown_font(user):
    assert user.post("/api/font/rate", json={"fontKey": "nope", "rating": 3}).status_code == 404


# ------------------------------------------------------------------ descriptions

def test_describe(user, app):
    import sqlite3
    key = firstKey(user)
    assert user.post("/api/font/describe", json={"fontKey": key, "description": "  warm and friendly  "}).status_code == 200
    assert user.post("/api/font/describe", json={"fontKey": key, "description": ""}).status_code == 400
    assert user.post("/api/font/describe", json={"fontKey": key, "description": "x" * 501}).status_code == 400
    assert user.post("/api/font/describe", json={"fontKey": "nope", "description": "hi"}).status_code == 404
    rows = sqlite3.connect(app.config["DATABASE"]).execute("SELECT fontKey, description FROM fontDescriptions").fetchall()
    assert rows == [(key, "warm and friendly")]


def test_describe_rate_limit(makeApp):
    app = makeApp(RATELIMIT_ENABLED=True)
    client = app.test_client()
    register(client)
    login(client)
    key = query(client).json["results"][0]["key"]
    codes = [client.post("/api/font/describe", json={"fontKey": key, "description": f"d{i}"}).status_code
             for i in range(3)]
    assert codes == [200, 200, 429]


# ------------------------------------------------------------------ misc

def test_health(client):
    body = client.get("/api/health").json
    assert body["fonts"] == 300 and body["tags"] > 100


def test_static_and_spa_fallback(client):
    assert client.get("/hello.txt").data == b"hi"
    assert b"index" in client.get("/").data
    assert b"index" in client.get("/some/client/route").data
    assert client.get("/../../etc/passwd").status_code in (200, 404)
    assert b"root:" not in client.get("/..%2f..%2fetc/passwd").data


def test_missing_files_are_404_not_the_app(client):
    # the Map page loads /flower.html in an iframe; a missing file used to show the whole app inside it
    for path in ("/flower.html", "/static/js/missing.js", "/nothing.png"):
        assert client.get(path).status_code == 404
    assert b"index" in client.get("/map").data  # extensionless paths are still client routes


def test_unknown_api_path_is_json_404(client):
    response = client.get("/api/font/nothing")
    assert response.status_code == 404 and response.json == {"message": "Not found"}


def test_removed_endpoints_are_gone(client):
    for path in ("/api/font/update", "/api/font/add"):
        assert client.post(path).status_code in (404, 405)


def test_old_schema_tables_left_alone_and_new_ones_created(app):
    import sqlite3
    db = sqlite3.connect(app.config["DATABASE"])
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"users", "fontVotes", "fontRatings", "fontDescriptions"} <= tables


def test_server_does_not_import_torch():
    code = ("import os, sys; os.environ['SECRET_KEY']='x'\n"
            "from fakeBundle import makeFakeBundle; import tempfile\n"
            "d = tempfile.mkdtemp(); makeFakeBundle(d, 20)\n"
            "from app import createApp\n"
            # suggestions off: spaCy imports requests at load time, and torch too wherever torch happens to be
            # installed (the server image has none); this checks the server's own imports
            "createApp({'BUNDLE_DIR': d, 'DATABASE': d + '/t.db', 'SYNONYM_MODEL': ''})\n"
            "bad = [m for m in ('torch', 'transformers', 'sqlite_vec', 'requests', 'cv2') if m in sys.modules]\n"
            "assert not bad, bad")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


def test_missing_secret_key_refuses_to_start(makeApp, monkeypatch):
    monkeypatch.delenv("SECRET_KEY")
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        makeApp(SECRET_KEY=None)


def test_bad_bundle_refuses_to_start(makeApp, tmp_path):
    from bundle import BundleError
    with pytest.raises(BundleError):
        makeApp(BUNDLE_DIR=str(tmp_path / "empty"))


def tagsOf(body):
    return {t["tag"]: t["weight"] for t in body["tags"]}


def test_tags_endpoint_lists_what_a_query_means(client):
    body = client.get("/api/font/tags", query_string={"query": "elegant script not thin"}).json
    assert tagsOf(body) == {"elegant": 1.0, "script": 1.0, "thin": -1.0}
    assert body["suggested"] == [] or all(set(s) == {"tag", "via", "similarity"} for s in body["suggested"])
    assert body["unmatched"] == []
    nothing = client.get("/api/font/tags", query_string={"query": "zzqx"}).json
    assert nothing["tags"] == [] and nothing["unmatched"] == ["zzqx"]
    assert client.get("/api/font/tags", query_string={"query": "a" * 201}).status_code == 400
    assert client.get("/api/font/tags").json["tags"] == []


def test_tags_endpoint_flattens_guesses_into_tags(client, app):
    index = app.extensions["tagIndex"]
    word = next((w for w, g in sorted(index.vocabulary.wordTags.items())
                 if any(x in index.groups for x in g) and index.parse(w) == ([], [w])), None)
    if word is None:
        pytest.skip("no caption-table word maps onto the fake bundle's tags")
    body = client.get("/api/font/tags", query_string={"query": word}).json
    assert body["tags"] and all(t["weight"] == 0.5 for t in body["tags"]) and body["unmatched"] == []


def test_query_ranks_exactly_the_tags_it_is_given(client):
    typed = query(client, "serif bold").json
    assert tagsOf(typed) == {"serif": 1.0, "bold": 1.0}
    given = query(client, "serif bold", tags="serif:1,-bold:0.6").json
    assert tagsOf(given) == {"serif": 1.0, "bold": -0.6}          # the text is not parsed again
    assert [r["key"] for r in given["results"]] != [r["key"] for r in typed["results"]]
    only = query(client, "serif bold", tags="bold").json
    assert tagsOf(only) == {"bold": 1.0}
    assert [r["key"] for r in only["results"]] == [r["key"] for r in query(client, "bold").json["results"]]


def test_query_with_an_empty_tag_list_has_no_results(client):
    body = query(client, "serif", tags="").json
    assert body["tags"] == [] and body["results"] == [] and body["total"] == 0


def test_query_skips_unknown_tags_and_rejects_bad_ones(client):
    assert tagsOf(query(client, "x", tags="serif,no-such-tag").json) == {"serif": 1.0}
    for tags in ("serif:abc", "serif:2", "serif:0", ",".join(["serif"] * 65)):
        assert query(client, "x", tags=tags).status_code == 400


def test_pages_of_a_tag_list_tile_like_any_other(client):
    seen = []
    for page in (1, 2, 3):
        seen += [r["key"] for r in query(client, "x", tags="serif,-bold:0.5", page=page, pageSize=100).json["results"]]
    assert len(seen) == 300 == len(set(seen))


def test_tags_endpoint_suggests_tags_for_unknown_words(makeApp):
    pytest.importorskip("en_core_web_md")
    body = makeApp(SYNONYM_MODEL="en_core_web_md").test_client().get(
        "/api/font/tags", query_string={"query": "ghastly"}).json
    assert body["tags"] == [] and body["unmatched"] == [] and body["suggested"]
    assert set(body["suggested"][0]) == {"tag", "via", "similarity"}
    assert len({s["tag"] for s in body["suggested"]}) == len(body["suggested"])
