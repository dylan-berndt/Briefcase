import sqlite3

from db import initializeDB, tagsKey

OLD_VOTES = '''
    CREATE TABLE users (id INTEGER PRIMARY KEY, publicID TEXT NOT NULL UNIQUE, username TEXT NOT NULL UNIQUE,
                        hash TEXT NOT NULL, admin INTEGER DEFAULT 0);
    CREATE TABLE fontVotes (
        userID INTEGER NOT NULL REFERENCES users(id), fontKey TEXT NOT NULL, query TEXT NOT NULL,
        vote INTEGER NOT NULL CHECK (vote IN (-1, 1)), created TEXT NOT NULL,
        PRIMARY KEY (userID, fontKey, query));
    CREATE INDEX fontVotesByFont ON fontVotes (fontKey);
'''


def makeOldDB(path):
    db = sqlite3.connect(path)
    db.executescript(OLD_VOTES)
    db.execute("INSERT INTO users VALUES (1, 'p', 'alice', 'h', 0)")
    db.executemany("INSERT INTO fontVotes VALUES (1, ?, ?, ?, ?)", [
        ("google:A", "serif", 1, "2026-09-01T00:00:00"), ("google:B", "serif", -1, "2026-09-02T00:00:00")])
    db.commit()
    db.close()


def test_old_votes_keep_their_query_and_get_empty_tags(tmp_path):
    path = str(tmp_path / "old.db")
    makeOldDB(path)
    initializeDB(path)
    db = sqlite3.connect(path)
    assert [r[1] for r in db.execute("PRAGMA table_info(fontVotes)")] == [
        "userID", "fontKey", "query", "tags", "vote", "created"]
    assert db.execute("SELECT fontKey, query, tags, vote, created FROM fontVotes ORDER BY fontKey").fetchall() == [
        ("google:A", "serif", "", 1, "2026-09-01T00:00:00"), ("google:B", "serif", "", -1, "2026-09-02T00:00:00")]
    # the key now includes the tags: the same font and query under other tags is a second vote
    db.execute("INSERT INTO fontVotes VALUES (1, 'google:A', 'serif', 'bold:1', -1, 'now')")
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
    assert "fontVotesByFont" in names and "fontVotesOld" not in names and "fontRatings" in names
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_migrating_twice_changes_nothing(tmp_path):
    path = str(tmp_path / "old.db")
    makeOldDB(path)
    initializeDB(path)
    initializeDB(path)
    db = sqlite3.connect(path)
    assert db.execute("SELECT COUNT(*) FROM fontVotes").fetchone()[0] == 2


def test_a_fresh_database_gets_the_new_table(tmp_path):
    path = str(tmp_path / "new.db")
    initializeDB(path)
    db = sqlite3.connect(path)
    assert "tags" in [r[1] for r in db.execute("PRAGMA table_info(fontVotes)")]


def test_tags_key_is_canonical():
    assert tagsKey([("serif", 1.0), ("bold", 0.5)]) == "bold:0.5,serif:1"
    assert tagsKey([("serif", 1.0), ("bold", 0.5)]) == tagsKey([("bold", 0.5), ("serif", 1.0)])
    assert tagsKey([("a", 0.33333)]) == "a:0.333" and tagsKey([]) == ""
