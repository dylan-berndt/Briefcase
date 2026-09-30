import math
import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import Flask, jsonify, request, g, make_response, send_from_directory, abort, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash

from bundle import Bundle
from db import initializeDB, normalizeQuery, MAX_QUERY
from tagsearch import TagIndex

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PAGE_SIZE = 24
MAX_PAGE_SIZE = 100
MAX_DESCRIPTION = 500
USERNAME = re.compile(r"[A-Za-z0-9_.\-]{3,32}")
MIN_PASSWORD = 8


def createApp(overrides=None):
    app = Flask(__name__, static_folder=None)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY"),
        DATABASE=os.getenv("SQLITE_PATH", "/data/fontsearch.db"),
        BUNDLE_DIR=os.getenv("BUNDLE_DIR", os.path.join(HERE, "data")),
        VERIFY_BUNDLE=os.getenv("VERIFY_BUNDLE") == "1",
        STATIC_DIR=os.getenv("STATIC_DIR", os.path.join(HERE, "static")),
        VOCABULARY=os.getenv("TAG_VOCABULARY"),
        COOKIE_SECURE=os.getenv("COOKIE_SECURE", "1") == "1",
        RATELIMIT_ENABLED=True,
    )
    app.config.update(overrides or {})
    if not app.config["SECRET_KEY"]:
        raise RuntimeError("SECRET_KEY is not set")

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["2000 per day", "500 per hour"],
        storage_uri="memory://",
    )
    app.limiter = limiter  # Flask-Limiter only keeps a weak reference to itself

    bundle = Bundle(app.config["BUNDLE_DIR"], verify=app.config["VERIFY_BUNDLE"])
    index = TagIndex(bundle, app.config["VOCABULARY"])
    specimenType = bundle.manifest["specimen"]["mimetype"]
    initializeDB(app.config["DATABASE"])

    app.extensions["bundle"] = bundle
    app.extensions["tagIndex"] = index

    # ------------------------------------------------------------ helpers

    def dbRequired(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "db" not in g:
                g.db = sqlite3.connect(app.config["DATABASE"])
                g.db.row_factory = sqlite3.Row

            cursor = g.db.cursor()
            try:
                response = f(cursor, *args, **kwargs)
                g.db.commit()
                return response
            except Exception:
                g.db.rollback()
                raise
            finally:
                cursor.close()

        return decorated

    def currentUser(cursor):
        """The logged-in user's row, or None for a missing, expired or invalid token."""
        token = request.cookies.get("token")
        if not token:
            return None
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        except jwt.InvalidTokenError:
            return None
        cursor.execute("SELECT id, publicID, username FROM users WHERE publicID = ?", (data.get("publicID"),))
        return cursor.fetchone()

    def loginRequired(f):
        @wraps(f)
        def decorated(cursor, *args, **kwargs):
            user = currentUser(cursor)
            if user is None:
                return jsonify({"message": "Not logged in"}), 401
            return f(cursor, user, *args, **kwargs)

        return decorated

    def now():
        return datetime.now(timezone.utc).isoformat()

    def bodyField(name):
        body = request.get_json(silent=True)
        return body.get(name) if isinstance(body, dict) else None

    def fontFromKey(key):
        if not isinstance(key, str) or key not in bundle.indexByKey:
            abort(make_response(jsonify({"message": "Font not found"}), 404))
        return key

    def specimenURL(i):
        return f"/api/font/specimen/{i}?v={bundle.version}"

    @app.teardown_appcontext
    def closeDB(exception):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    # ------------------------------------------------------------ accounts

    @app.route("/api/font/register", methods=["POST"])
    @limiter.limit("3 per day")
    @dbRequired
    def register(cursor):
        username, password = request.form.get("username", ""), request.form.get("password", "")
        if not USERNAME.fullmatch(username):
            return jsonify({"message": "Username must be 3-32 letters, digits, dots, dashes or underscores."}), 400
        if len(password) < MIN_PASSWORD or len(password) > 128:
            return jsonify({"message": f"Password must be {MIN_PASSWORD}-128 characters."}), 400

        cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return jsonify({"message": "User already exists. Please login."}), 400

        cursor.execute("INSERT INTO users (publicID, username, hash) VALUES (?, ?, ?)",
                       (str(uuid.uuid4()), username, generate_password_hash(password)))
        return jsonify({"message": "Registered successfully"}), 200

    @app.route("/api/font/login", methods=["POST"])
    @limiter.limit("5 per hour")
    @dbRequired
    def login(cursor):
        username, password = request.form.get("username", ""), request.form.get("password", "")

        cursor.execute("SELECT publicID, hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user is None or not check_password_hash(user["hash"], password):
            return jsonify({"message": "Invalid username or password."}), 400

        token = jwt.encode({"publicID": user["publicID"], "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                           app.config["SECRET_KEY"], algorithm="HS256")
        response = make_response(jsonify({"message": "Logged in successfully", "username": username}), 200)
        response.set_cookie("token", token, httponly=True, secure=app.config["COOKIE_SECURE"],
                            samesite="Strict", max_age=3600)
        return response

    @app.route("/api/font/logout", methods=["POST"])
    def logout():
        response = make_response(jsonify({"message": "Logged out"}), 200)
        response.delete_cookie("token", samesite="Strict", secure=app.config["COOKIE_SECURE"])
        return response

    @app.route("/api/font/me", methods=["GET"])
    @dbRequired
    def me(cursor):
        user = currentUser(cursor)
        return jsonify({"username": user["username"] if user else None}), 200

    # ------------------------------------------------------------ search

    @app.route("/api/font/query", methods=["GET"])
    @limiter.limit("120 per minute")
    @dbRequired
    def findFonts(cursor):
        query = request.args.get("query", "")
        if len(query) > MAX_QUERY:
            return jsonify({"message": f"Query is longer than {MAX_QUERY} characters"}), 400
        try:
            page = int(request.args.get("page", 1))
            pageSize = int(request.args.get("pageSize", DEFAULT_PAGE_SIZE))
        except ValueError:
            return jsonify({"message": "page and pageSize must be integers"}), 400
        if page < 1 or not 1 <= pageSize <= MAX_PAGE_SIZE:
            return jsonify({"message": f"page must be >= 1 and pageSize 1-{MAX_PAGE_SIZE}"}), 400

        order, terms, unmatched = index.search(query)
        total = len(order)
        ids = [int(i) for i in order[(page - 1) * pageSize: page * pageSize]]
        keys = [bundle.fonts[i]["key"] for i in ids]

        ratings, mine, votes = {}, {}, {}
        if keys:
            marks = ",".join("?" * len(keys))
            cursor.execute(f"SELECT fontKey, AVG(rating) AS average, COUNT(*) AS count FROM fontRatings "
                           f"WHERE fontKey IN ({marks}) GROUP BY fontKey", keys)
            ratings = {row["fontKey"]: row for row in cursor.fetchall()}
            user = currentUser(cursor)
            if user is not None:
                cursor.execute(f"SELECT fontKey, rating FROM fontRatings WHERE userID = ? AND fontKey IN ({marks})",
                               [user["id"], *keys])
                mine = {row["fontKey"]: row["rating"] for row in cursor.fetchall()}
                cursor.execute(f"SELECT fontKey, vote FROM fontVotes WHERE userID = ? AND query = ? "
                               f"AND fontKey IN ({marks})", [user["id"], normalizeQuery(query), *keys])
                votes = {row["fontKey"]: row["vote"] for row in cursor.fetchall()}

        results = []
        for i, key in zip(ids, keys):
            font = bundle.fonts[i]
            rating = ratings.get(key)
            results.append({
                "key": key,
                "name": font["name"],
                "source": font["source"],
                "url": font["url"],
                "creator": font.get("creator"),
                "specimen": specimenURL(i),
                "rating": {"average": round(rating["average"], 2) if rating else None,
                           "count": rating["count"] if rating else 0,
                           "mine": mine.get(key)},
                "vote": votes.get(key, 0),
            })

        return jsonify({
            "results": results,
            "page": page,
            "pageSize": pageSize,
            "total": total,
            "totalPages": math.ceil(total / pageSize),
            "tags": [{"tag": name, "weight": round(float(weight), 3)} for name, weight in terms],
            "unmatched": unmatched,
        }), 200

    @app.route("/api/font/specimen/<int:i>", methods=["GET"])
    @limiter.exempt
    def specimen(i):
        if not 0 <= i < len(bundle.fonts):
            abort(404)
        response = Response(bundle.specimen(i), mimetype=specimenType)
        # the ?v= in the URL changes with the bundle, so a cached copy is never stale
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        response.set_etag(f"{bundle.version}-{i}")
        return response.make_conditional(request)

    # ------------------------------------------------------------ feedback

    @app.route("/api/font/approve", methods=["POST"])
    @limiter.limit("60 per minute")
    @dbRequired
    @loginRequired
    def approveFont(cursor, user):
        """Does this font answer this query? vote is 1 (yes), -1 (no) or 0 (clear the user's vote)."""
        fontKey = fontFromKey(bodyField("fontKey"))
        query = bodyField("query")
        vote = bodyField("vote")
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY:
            return jsonify({"message": "Invalid query"}), 400
        if vote not in (-1, 0, 1) or isinstance(vote, bool):
            return jsonify({"message": "vote must be 1, -1 or 0"}), 400
        query = normalizeQuery(query)

        if vote == 0:
            cursor.execute("DELETE FROM fontVotes WHERE userID = ? AND fontKey = ? AND query = ?",
                           (user["id"], fontKey, query))
        else:
            cursor.execute("INSERT INTO fontVotes (userID, fontKey, query, vote, created) VALUES (?, ?, ?, ?, ?) "
                           "ON CONFLICT (userID, fontKey, query) DO UPDATE SET vote = excluded.vote, "
                           "created = excluded.created", (user["id"], fontKey, query, vote, now()))
        return jsonify({"message": "Successful", "vote": vote}), 200

    @app.route("/api/font/rate", methods=["POST"])
    @limiter.limit("60 per minute")
    @dbRequired
    @loginRequired
    def rateFont(cursor, user):
        """Is this a good font? rating is 1-5 stars, or 0 to clear the user's rating."""
        fontKey = fontFromKey(bodyField("fontKey"))
        rating = bodyField("rating")
        if rating not in (0, 1, 2, 3, 4, 5) or isinstance(rating, bool):
            return jsonify({"message": "rating must be an integer from 0 to 5"}), 400

        if rating == 0:
            cursor.execute("DELETE FROM fontRatings WHERE userID = ? AND fontKey = ?", (user["id"], fontKey))
        else:
            cursor.execute("INSERT INTO fontRatings (userID, fontKey, rating, created) VALUES (?, ?, ?, ?) "
                           "ON CONFLICT (userID, fontKey) DO UPDATE SET rating = excluded.rating, "
                           "created = excluded.created", (user["id"], fontKey, rating, now()))

        cursor.execute("SELECT AVG(rating) AS average, COUNT(*) AS count FROM fontRatings WHERE fontKey = ?", (fontKey,))
        row = cursor.fetchone()
        return jsonify({"message": "Successful",
                        "rating": {"average": round(row["average"], 2) if row["count"] else None,
                                   "count": row["count"], "mine": rating or None}}), 200

    @app.route("/api/font/describe", methods=["POST"])
    @limiter.limit("2 per minute")
    @dbRequired
    @loginRequired
    def describeFont(cursor, user):
        fontKey = fontFromKey(bodyField("fontKey"))
        description = bodyField("description")
        if not isinstance(description, str) or not description.strip() or len(description) > MAX_DESCRIPTION:
            return jsonify({"message": f"Description must be 1-{MAX_DESCRIPTION} characters"}), 400

        cursor.execute("INSERT INTO fontDescriptions (fontKey, description, userID, created) VALUES (?, ?, ?, ?)",
                       (fontKey, description.strip(), user["id"], now()))
        return jsonify({"message": "Successful"}), 200

    @app.route("/api/health", methods=["GET"])
    @limiter.exempt
    def health():
        return jsonify({"fonts": len(bundle.fonts), "tags": len(bundle.vocab), "version": bundle.version}), 200

    # ------------------------------------------------------------ frontend

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    @limiter.exempt
    def serve(path):
        if path.startswith("api/"):
            abort(404)
        staticDir = app.config["STATIC_DIR"]
        if path != "" and os.path.isfile(os.path.join(staticDir, path)):
            return send_from_directory(staticDir, path)
        # A missing file (the Map page's iframe, say) is a 404, not the app; only extensionless paths are routes
        if os.path.splitext(path)[1]:
            abort(404)
        return send_from_directory(staticDir, "index.html")

    @app.errorhandler(404)
    def notFound(error):
        if request.path.startswith("/api/"):
            return jsonify({"message": "Not found"}), 404
        return error

    return app


def __getattr__(name):
    # `gunicorn app:app` builds the app on first access. Importing this module for createApp (the tests) does not.
    if name == "app":
        global app
        app = createApp()
        return app
    raise AttributeError(name)


if __name__ == "__main__":
    createApp().run(host="0.0.0.0", port=8000)
