from flask import Flask, jsonify, request, abort

import json
import os
from io import BytesIO

import hashlib
import secrets
from urllib.parse import urlparse
import requests

import sqlite3
import sqlite_vss

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import *

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

# TODO: Add user description field, slowly improve model
cursor.execute('''
    CREATE TABLE IF NOT EXISTS descriptions (
        FOREIGN KEY (fontID) REFERENCES fonts (id),
        description TEXT NOT NULL
    )
''')

conn.commit()


def hashPassword(password, salt=None, iterations=260000):
    if salt is None:
        salt = secrets.token_hex(16)
    hashData = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return salt, hashData.hex()


# TODO: Probably change this, use JWT
with open("password.txt", "r") as file:
    data = file.read()
if data == "":
    salt, hashword = hashPassword(input("Create Admin Password: "), None)
    with open("password.txt", "w") as file:
        file.write(f"{salt}${hashword}")
else:
    salt, hashword = data.split("$")


def checkPassword():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        password = auth.split(" ", 1)[1].strip()
        _, hashed = hashPassword(password, salt)
        if secrets.compare_digest(hashword, hashed):
            return
        
        abort(401)

    abort(401)


@app.route('/api/font/query', methods=['GET'])
def findFonts():
    query, includePaid = request.args.get("query", ""), request.args.get("includePaid", True)
    query = "a " + query + " font"
    tokens = Description.tokenizer([query], padding=False, return_tensors="pt")
    with torch.no_grad():
        output = textModel(**tokens).pooler_output

    rows = cursor.fetchall(f'''
        SELECT name, distance, location, file FROM fonts WHERE (? OR NOT paid) AND vss_search(embedding, ?) LIMIT 20
    ''', (includePaid, output.numpy().tolist()))

    results = [dict(zip(["name", "score", "file", "url"], rows[i])) for i in range(len(rows))]

    return jsonify({"results": results}), 200


@app.route('/api/font/update', methods=['POST'])
def updateRegistry():
    checkPassword()

    registered = cursor.fetchall("SELECT * FROM registry")

    for row in registered:
        id, name, location, file, paid = row

        response = requests.get(file)

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
def addFontToRegistry():
    checkPassword()

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


@app.route('/api/font/change/', methods=['POST'])
def changeAdminPassword():
    checkPassword()

    newPassword = request.headers.get("Password", "")
    if len(newPassword) < 12 or len(newPassword) > 48 or "\n" in newPassword:
        abort(400)

    global salt, hashword
    salt, hashword = hashPassword(newPassword)
    with open("password.txt", "w") as file:
        file.write(f"{salt}${hashword}")


if __name__ == '__main__':
    app.run(port=5000)


