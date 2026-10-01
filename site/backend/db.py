import re
import sqlite3

# Fonts live in the bundle, not in the database, so votes, ratings and descriptions refer to a font by its bundle key.
# The pre-tag-search schema (vec0 `fonts`, `fontsMeta`, `registry`, `ratings`, `approvals`, `descriptions`) is left
# alone: its tables cannot be dropped without the sqlite-vec extension, and none of the endpoints that wrote to
# them ever worked, so they hold nothing but `users` worth keeping.
SCHEMA = '''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        publicID TEXT NOT NULL UNIQUE,
        username TEXT NOT NULL UNIQUE,
        hash TEXT NOT NULL,
        admin INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS fontVotes (
        userID INTEGER NOT NULL REFERENCES users(id),
        fontKey TEXT NOT NULL,
        query TEXT NOT NULL,
        tags TEXT NOT NULL DEFAULT '',
        vote INTEGER NOT NULL CHECK (vote IN (-1, 1)),
        created TEXT NOT NULL,
        PRIMARY KEY (userID, fontKey, query, tags)
    );
    CREATE INDEX IF NOT EXISTS fontVotesByFont ON fontVotes (fontKey);

    CREATE TABLE IF NOT EXISTS fontRatings (
        userID INTEGER NOT NULL REFERENCES users(id),
        fontKey TEXT NOT NULL,
        rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
        created TEXT NOT NULL,
        PRIMARY KEY (userID, fontKey)
    );
    CREATE INDEX IF NOT EXISTS fontRatingsByFont ON fontRatings (fontKey);

    CREATE TABLE IF NOT EXISTS fontDescriptions (
        id INTEGER PRIMARY KEY,
        fontKey TEXT NOT NULL,
        description TEXT NOT NULL,
        userID INTEGER NOT NULL REFERENCES users(id),
        created TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS fontDescriptionsByFont ON fontDescriptions (fontKey);
'''

MAX_QUERY = 200


def addTagsToVotes(conn):
    """A vote is now filed under the query and the tags the results were ranked on. Databases from before that have
    fontVotes without the column (and with a primary key that cannot hold it), so the table is rebuilt; the old votes
    keep their query and get tags = '' (unknown)."""
    columns = [row[1] for row in conn.execute("PRAGMA table_info(fontVotes)")]
    if not columns or "tags" in columns:
        return
    conn.execute("ALTER TABLE fontVotes RENAME TO fontVotesOld")
    conn.execute("DROP INDEX IF EXISTS fontVotesByFont")
    conn.executescript(SCHEMA)  # creates the new fontVotes and the other tables if missing
    conn.execute("INSERT INTO fontVotes (userID, fontKey, query, tags, vote, created) "
                 "SELECT userID, fontKey, query, '', vote, created FROM fontVotesOld")
    conn.execute("DROP TABLE fontVotesOld")


def initializeDB(path):
    conn = sqlite3.connect(path)
    addTagsToVotes(conn)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def tagsKey(terms):
    """[(tag, weight)] -> the canonical string a vote stores: "bold:1,script:0.5", sorted by tag."""
    return ",".join(f"{name}:{round(float(weight), 3):g}" for name, weight in sorted(terms))


def normalizeQuery(query):
    """What a vote is filed under, so "Bold  Script" and "bold script" are the same search."""
    return re.sub(r"\s+", " ", query.strip().lower())[:MAX_QUERY]
