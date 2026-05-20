from flask import Flask, jsonify, request, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import g, send_from_directory

import json
import os
from io import BytesIO
from functools import wraps

from urllib.parse import urlparse
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import uuid
import requests

import sqlite3
import sqlite_vec

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils import *

from datetime import datetime, timezone, timedelta

testDir = os.path.join("checkpoints", "finetune", "best")
print(os.path.exists(testDir))
textModel = CLIPTextModel.from_pretrained(os.path.join(testDir, "text"), local_files_only=True)
Description.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

imageModel, conf = UNet.load(testDir, name="image")

textModel.eval()
imageModel.eval()

freeDomains = ["www.dafont.com", "fonts.google.com", "www.fontsquirrel.com"]
paidDomains = ["www.myfonts.com"]

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ["SECRET_KEY"]
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

DATABASE = os.getenv("SQLITE_PATH", "/data/fontsearch.db")


def initializeDB():
    if not os.path.exists(DATABASE):
        conn = sqlite3.connect(DATABASE)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        cursor = conn.cursor()

        cursor.execute(f'''
            CREATE VIRTUAL TABLE IF NOT EXISTS fonts USING vec0(
                id INTEGER PRIMARY KEY,
                embedding({conf.model.textProjection}),
                name TEXT NOT NULL UNIQUE,
                location TEXT NOT NULL,
                file TEXT NOT NULL,
                paid INTEGER DEFAULT 0
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS registry (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT NOT NULL,
                file TEXT NOT NULL,
                paid INTEGER DEFAULT 0
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                publicID TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL UNIQUE,
                hash TEXT NOT NULL,
                admin INTEGER DEFAULT 0
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS descriptions (
                FOREIGN KEY (fontID) REFERENCES fonts (id),
                description TEXT NOT NULL,
                FOREIGN KEY (userID) REFERENCES users (id),
                created TEXT NOT NULL
            )
        ''')

        conn.commit()
        cursor.close()
        conn.close()


def dbRequired(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "db" not in g:
            g.db = sqlite3.connect(DATABASE)

            g.db.enable_load_extension(True)
            sqlite_vec.load(g.db)
            g.db.enable_load_extension(False)

            g.db.row_factory = sqlite3.Row
        
        cursor = g.db.cursor()

        try:
            response = f(cursor, *args, **kwargs)
            g.db.commit()
            return response
        except:
            g.db.rollback()
            raise
        finally:
            cursor.close()

    return decorated


def loginRequired(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('token')

        if not token:
            return jsonify({'message': 'Not logged in'}), 401
        
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            cursor = g.db.cursor()
            cursor.execute("SELECT * FROM users WHERE publicID = ?", (data['publicID'],))
            user = cursor.fetchone()

            if not user:
                return jsonify({"message": "Not logged in"}), 401

        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Session expired"}), 401

        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401
    
        return f(user, *args, **kwargs)

    return decorated


def adminRequired(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('token')

        if not token:
            return jsonify({'message': 'Not logged in'}), 401
        
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            cursor = g.db.cursor()
            cursor.execute("SELECT username, publicID, admin FROM users WHERE publicID = ?", (data['publicID'],))
            user = cursor.fetchone()

            if not user[2]:
                return jsonify({'message': 'Not admin'}), 403

        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Session expired"}), 401

        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401
        
        finally:
            cursor.close()
    
        return f(user, *args, **kwargs)
    return decorated


@app.teardown_appcontext
def closeDB(exception):
    db = g.pop("db", None)

    if db is not None:
        db.close()


# TODO: Enforce password length, characters, etc.
@app.route('/api/font/register', methods=['POST'])
@limiter.limit("3 per day")
@dbRequired
def register(cursor):
    username, password = request.form['username'], request.form['password']

    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    users = cursor.fetchall()
    existing = len(users) > 0
    if existing:
        return jsonify({'message': 'User already exists. Please login.'}), 400
    
    hashed = generate_password_hash(password)
    cursor.execute("INSERT INTO users (publicID, username, hash) VALUES (?, ?, ?)", (str(uuid.uuid4()), username, hashed))

    return jsonify({'message': 'Registered successfully'}), 200


@app.route('/api/font/login', methods=['POST'])
@limiter.limit("5 per hour")
@dbRequired
def login(cursor):
    username, password = request.form['username'], request.form['password']

    cursor.execute("SELECT publicID, username, hash FROM users WHERE username = ?", (username,))
    users = cursor.fetchall()
    if len(users) == 0:
        return jsonify({'message': 'Invalid username or password.'}), 400
    
    user = users[0]
    publicID, name, hash = user
    if not check_password_hash(hash, password):
        return jsonify({'message': 'Invalid username or password.'}), 400
    
    token = jwt.encode({'publicID': publicID, 'exp': datetime.now(timezone.utc) + timedelta(hours=1)}, app.config["SECRET_KEY"], algorithm="HS256")

    response = Flask.make_response(jsonify({'message': 'Logged in successfully'}), 200)
    response.set_cookie('token', token, httponly=True, secure=True, samesite="Strict")

    return response


@app.route('/api/font/query', methods=['GET'])
@limiter.limit("40 per day")
@dbRequired
def findFonts(cursor):
    query, includePaid = request.args.get("query", ""), request.args.get("includePaid", True)
    query = "a " + query + " font"
    tokens = Description.tokenizer([query], padding=False, return_tensors="pt")
    with torch.no_grad():
        output = textModel(**tokens).pooler_output

    cursor.execute(f'''
        SELECT name, distance, location, file FROM fonts WHERE (? OR NOT paid) AND embedding match ? ORDER BY distance LIMIT 20
    ''', (includePaid, output.numpy().tolist()))
    rows = cursor.fetchall()

    results = [dict(zip(["name", "score", "file", "url"], rows[i])) for i in range(len(rows))]

    return jsonify({"results": results}), 200


@app.route('/api/font/describe', methods=['POST'])
@limiter.limit("2 per minute")
@dbRequired
@loginRequired
def describeFont(cursor, user):
    fontName = request.args.get("fontName", "")
    cursor.execute('''
        SELECT id FROM fonts WHERE name = ?
    ''', (fontName,))
    rows = cursor.fetchall()

    if len(rows) == 0:
        return jsonify({'message': 'Font not found'}), 401
    
    description = request.args.get("description", "")
    if not description:
        return jsonify({'message': 'Invalid description'}), 401
    
    cursor.execute('''
        INSERT INTO descriptions (fontID, description, userID, created) VALUES (?, ?, ?, ?)
    ''', (rows[0][0], description, user[0], datetime.now(timezone.utc)))

    return jsonify({'message': 'Successful'}), 200


@app.route('/api/font/update', methods=['POST'])
@dbRequired
@adminRequired
def updateRegistry(cursor):
    cursor.execute("SELECT * FROM registry")
    registered = cursor.fetchall()

    for row in registered:
        id, name, location, file, paid = row

        response = requests.get(file, allow_redirects=False, timeout=5, stream=True)

        if not response.ok:
            abort(500)

        data = BytesIO(response.content)

        font, fontName, fontStyle, images = imagesFromFont(data, conf.fontSize, int(conf.fontSize * 1.5), chars=latin)
        images = torch.stack([torch.tensor(np.array(image) / 255, dtype=torch.float32).unsqueeze(-1) for image in images], dim=0)
        
        embeddings = imageModel(images)
        embeddings = torch.linalg.norm(embeddings, axis=1)
        embedding = torch.mean(embeddings, dim=0).numpy().tolist()

        cursor.execute(f"INSERT INTO fonts (embedding, name, location, file, paid) VALUES (?, ?, ?, ?, ?)",
                       (embedding, name, location, file, paid))
        cursor.execute(f"DELETE FROM registry WHERE id = ?", (id,))
    
    return jsonify({'message': 'Successful'}), 200


def checkDomain(url):
    url = urlparse(url)
    allowedHosts = freeDomains + paidDomains

    if url.scheme != "https":
        return False, False
    
    if url.hostname not in allowedHosts:
        return False, False

    return url.netloc in allowedHosts, url.netloc in paidDomains


@app.route('/api/font/add', methods=['POST'])
@limiter.exempt
@dbRequired
@adminRequired
def addFontToRegistry(cursor, user):
    name, url, file = request.args.get("name", ""), request.args.get("url", ""), request.args.get("file", "")
    if name == "" or url == "" or file == "":
        abort(400)

    urlGood = False
    fileGood = False
    paid = False

    urlGood, paid = checkDomain(url)
    fileGood, _ = checkDomain(file)

    fileGood = fileGood and (file.endswith(".otf") or file.endswith(".ttf"))

    if not urlGood or not fileGood:
        abort(400)
    
    cursor.execute(f'''
        INSERT INTO registry (name, location, paid, file) VALUES (?, ?, ?, ?)
    ''', (name, url, paid, file))


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(f"static/{path}"):
        return send_from_directory("static", path)
    return send_from_directory("static", "index.html")


if __name__ == '__main__':
    initializeDB()
    app.run(host="0.0.0.0", port=5000)


