"""A small cached client for the Datamuse API (https://www.datamuse.com/api/), for the offline builds in site/tools.

Nothing on the server uses it: buildSynonyms.py reads Datamuse once, offline, and writes configs/synonymTags.json. Datamuse
gives no bulk export and its terms say nothing about caching, so every answer is kept in a local sqlite file. A build that
is stopped carries on where it left off, and a rebuild from the same cache needs no network (offline=True). Please keep
the request rate modest; "requests may be rate-limited without notice" and the free tier is 100,000 a day.

    client = Datamuse("build/datamuse.sqlite")
    client.get("ml", "wet", 30)            # ['moisture', 'moist', 'damp', ...]   words meaning like "wet"
    client.get("rel_jjb", "cheese", 40)    # adjectives often used to modify "cheese"
    client.get("pos", "wet")               # ['adj', 'n', 'v']                    parts of speech of the word itself
    client.prefetch([("ml", "wet", 30), ("rel_ant", "wet", 12)])   # many at once, on a few threads
"""
import json
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://api.datamuse.com/words"
USER_AGENT = "briefcase-font-search/site-tools (offline synonym table build)"
RETRY_STATUS = {429, 500, 502, 503, 504}


class DatamuseError(RuntimeError):
    """A request that failed for good (not cached, so a rerun tries it again)."""


class Datamuse:
    def __init__(self, cachePath, offline=False, workers=6, perSecond=20.0, retries=5, timeout=20, opener=urllib.request.urlopen,
                 sleep=time.sleep):
        """offline: answer from the cache only. perSecond: cap on requests across all threads. opener and sleep are
        there for tests."""
        self.offline, self.workers, self.retries, self.timeout = offline, workers, retries, timeout
        self.opener, self.sleep, self.interval = opener, sleep, 1.0 / perSecond
        directory = os.path.dirname(cachePath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.db = sqlite3.connect(cachePath, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.lock = threading.RLock()
        self.nextAt = 0.0
        self.pending = 0
        self.requests = 0                # network requests made by this client

    # ---- cache

    @staticmethod
    def key(rel, word, n):
        return f"{rel}\t{n}\t{word}"

    def cached(self, rel, word, n=30):
        """The stored answer, or None when this question has not been asked."""
        with self.lock:
            row = self.db.execute("SELECT value FROM responses WHERE key = ?", (self.key(rel, word, n),)).fetchone()
        return json.loads(row[0]) if row else None

    def store(self, rel, word, n, value):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (self.key(rel, word, n), json.dumps(value)))
            self.pending += 1
            if self.pending >= 200:
                self.flush()

    def flush(self):
        with self.lock:
            self.db.commit()
            self.pending = 0

    def close(self):
        with self.lock:
            self.db.commit()
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- questions

    def get(self, rel, word, n=30):
        """The words Datamuse relates to `word` by `rel` (ml, rel_syn, rel_ant, rel_jjb, ...), best first, at most n.
        rel "pos" is instead the word's own parts of speech (n, adj, v, adv). Answers from the cache first; None when
        offline and not cached. Raises DatamuseError when the request fails."""
        found = self.cached(rel, word, n)
        if found is not None or self.offline:
            return found
        value = self.fetch(rel, word, n)
        self.store(rel, word, n, value)
        return value

    def fetch(self, rel, word, n):
        if rel == "pos":
            params = {"sp": word, "md": "p", "max": 1}
        else:
            params = {rel: word, "max": n}
        rows = self.request(params)
        if rel == "pos":
            # sp= is a spelling pattern, so check the answer is the word itself
            row = rows[0] if rows and rows[0].get("word") == word else {}
            return [t for t in row.get("tags", []) if t in ("n", "adj", "v", "adv")]
        return [row["word"] for row in rows if "word" in row]

    def request(self, params):
        url = URL + "?" + urllib.parse.urlencode(params)
        last = None
        for attempt in range(self.retries):
            self.throttle()
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with self.opener(request, timeout=self.timeout) as response:
                    self.requests += 1
                    data = json.loads(response.read().decode("utf-8"))
                if not isinstance(data, list):
                    raise DatamuseError(f"unexpected answer for {params}: {str(data)[:80]}")
                return data
            except urllib.error.HTTPError as error:
                last = error
                if error.code not in RETRY_STATUS:
                    raise DatamuseError(f"{params}: HTTP {error.code}") from error
                wait = float(error.headers.get("Retry-After", 0) or 0) if error.headers else 0
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as error:
                last, wait = error, 0
            self.sleep(max(wait, min(2 ** attempt, 30)))
        raise DatamuseError(f"{params}: gave up after {self.retries} tries ({last})")

    def throttle(self):
        with self.lock:
            now = time.monotonic()
            wait = self.nextAt - now
            self.nextAt = max(now, self.nextAt) + self.interval
        if wait > 0:
            self.sleep(wait)

    # ---- many at once

    def prefetch(self, questions, label="datamuse", every=2000):
        """Make sure every (rel, word, n) is cached, on a few threads. Returns the questions that failed (or, when
        offline, that are not cached)."""
        todo = [q for q in dict.fromkeys(questions) if self.cached(*q) is None]
        if not todo:
            return []
        if self.offline:
            return todo
        failed, done = [], 0
        print(f"{label}: {len(todo)} requests", file=sys.stderr)
        with ThreadPoolExecutor(self.workers) as pool:
            futures = {pool.submit(self.get, *q): q for q in todo}
            for future in as_completed(futures):
                done += 1
                try:
                    future.result()
                except DatamuseError as error:
                    failed.append(futures[future])
                    if len(failed) <= 5:
                        print(f"{label}: {error}", file=sys.stderr)
                if done % every == 0:
                    print(f"{label}: {done}/{len(todo)}", file=sys.stderr)
        self.flush()
        if failed:
            print(f"{label}: {len(failed)} requests failed; rerun to retry them", file=sys.stderr)
        return failed

