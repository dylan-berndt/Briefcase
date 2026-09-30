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
        vote INTEGER NOT NULL CHECK (vote IN (-1, 1)),
        created TEXT NOT NULL,
        PRIMARY KEY (userID, fontKey, query)
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


def initializeDB(path):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def normalizeQuery(query):
    """What a vote is filed under, so "Bold  Script" and "bold script" are the same search."""
    return re.sub(r"\s+", " ", query.strip().lower())[:MAX_QUERY]
