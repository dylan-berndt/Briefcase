from flask import Flask, jsonify, request, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

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
import sqlite_vss

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import *

from datetime import datetime, timezone, timedelta

torch.set_default_device("cpu")

testDir = os.path.join("..", "..", "checkpoints", "finetune", "best")
# print(os.path.exists(testDir))
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

conn = sqlite3.connect("fontsearch.db")
conn.enable_load_extension(True)
sqlite_vss.load(conn)
conn.enable_load_extension(False)
cursor = conn.cursor()

cursor.execute(f'''
    CREATE VIRTUAL TABLE IF NOT EXISTS fonts USING vss0(
        id INTEGER PRIMARY KEY,
        embedding({conf.model.textProjection}),
        name TEXT NOT NULL,
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

# TODO: Add user description field, slowly improve model
cursor.execute('''
    CREATE TABLE IF NOT EXISTS descriptions (
        FOREIGN KEY (fontID) REFERENCES fonts (id),
        description TEXT NOT NULL,
        FOREIGN KEY (userID) REFERENCES users (id),
        created TEXT NOT NULL
    )
''')

conn.commit()


@app.route('/api/font/register', methods=['POST'])
@limiter.limit("3 per day")
def register():
    username, password = request.form['username'], request.form['password']

    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    users = cursor.fetchall()
    existing = len(users) > 0
    if existing:
        return jsonify({'message': 'User already exists. Please login.'}), 400
    
    hashed = generate_password_hash(password)
    cursor.execute("INSERT INTO users (publicID, username, hash) VALUES (?, ?, ?)", (str(uuid.uuid4()), username, hashed))
    conn.commit()

    return jsonify({'message': 'Registered successfully'}), 200


@app.route('/api/font/login', methods=['POST'])
@limiter.limit("5 per hour")
def login():
    username, password = request.form['username'], request.form['password']

    cursor.execute("SELECT publicID, username, hash FROM users WHERE username = ?", (username,))
    users = cursor.fetchall()
    if len(users) == 0:
        return jsonify({'message': 'User does not exist.'}), 400
    
    user = users[0]
    publicID, name, hash = user
    if not check_password_hash(hash, password):
        return jsonify({'message': 'Incorrect password'}), 400
    
    token = jwt.encode({'publicID': publicID, 'expiration': datetime.now(timezone.utc) + timedelta(hours=1)}, app.config["SECRET_KEY"], algorithm="HS256")

    response = Flask.make_response(Flask.redirect(Flask.url_for("index")))
    response.set_cookie('token', token, httponly=True, secure=True, samesite="Strict")

    return response


def loginRequired(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('token')

        if not token:
            return jsonify({'message': 'Not logged in'}), 401
        
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            cursor.execute("SELECT * FROM users WHERE publicID = ?", (data['publicID'],))
            users = cursor.fetchall()
            user = users[0]

            expiration = datetime.fromtimestamp(data['expiration'], tz=timezone.utc)
            if expiration < datetime.now(timezone.utc):
                return jsonify({'message': 'Not logged in'}), 401

        except:
            return jsonify({'message': 'Not logged in'}), 401
    
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
            cursor.execute("SELECT username, publicID, admin FROM users WHERE publicID = ?", (data['publicID'],))
            users = cursor.fetchall()
            user = users[0]

            expiration = datetime.fromtimestamp(data['expiration'], tz=timezone.utc)
            if expiration < datetime.now(timezone.utc):
                return jsonify({'message': 'Not logged in'}), 401

            if not user[2]:
                return jsonify({'message': 'Not admin'}), 403

        except:
            return jsonify({'message': 'Not logged in'}), 401
    
        return f(user, *args, **kwargs)
    return decorated


@app.route('/api/font/query', methods=['GET'])
@limiter.limit("40 per day")
def findFonts():
    query, includePaid = request.args.get("query", ""), request.args.get("includePaid", True)
    query = "a " + query + " font"
    tokens = Description.tokenizer([query], padding=False, return_tensors="pt")
    with torch.no_grad():
        output = textModel(**tokens).pooler_output

    cursor.execute(f'''
        SELECT name, distance, location, file FROM fonts WHERE (? OR NOT paid) AND vss_search(embedding, ?) LIMIT 20
    ''', (includePaid, output.numpy().tolist()))
    rows = cursor.fetchall()

    results = [dict(zip(["name", "score", "file", "url"], rows[i])) for i in range(len(rows))]

    return jsonify({"results": results}), 200


@app.route('/api/font/describe', methods=['POST'])
@limiter.limit("2 per minute")
@loginRequired
def describeFont(user):
    pass


@app.route('/api/font/update', methods=['POST'])
@adminRequired
def updateRegistry():
    cursor.execute("SELECT * FROM registry")
    registered = cursor.fetchall()

    for row in registered:
        id, name, location, file, paid = row

        response = requests.get(file, allow_redirects=False)

        if not response.ok:
            abort(500)

        data = BytesIO(response.content)

        images = list(imagesFromFont(data, conf.fontSize, int(conf.fontSize * 1.5), chars=latin))
        images = torch.stack([torch.tensor(np.array(image) / 255, dtype=torch.float32).unsqueeze(-1) for image in images], dim=0)
        
        embeddings = imageModel(images)
        embeddings = torch.linalg.norm(embeddings, axis=1)
        embedding = torch.mean(embeddings, dim=0).numpy().tolist()

        cursor.execute(f"INSERT INTO fonts (embedding, name, location, file, paid) VALUES (?, ?, ?, ?, ?)",
                       (embedding, name, location, file, paid))
        cursor.execute(f"DELETE FROM registry WHERE id = {id}")
        conn.commit()
    
    return jsonify("Successful"), 200


def checkDomain(url):
    url = urlparse(url)
    urlDomain = url.netloc
    return urlDomain in (freeDomains + paidDomains), urlDomain in paidDomains


@app.route('/api/font/add', methods=['POST'])
@limiter.exempt
@adminRequired
def addFontToRegistry():
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
        INSERT INTO registry (name, location, paid) VALUES (?, ?, ?)
    ''', (name, url, paid))
    conn.commit()


if __name__ == '__main__':
    app.run(port=5000)


