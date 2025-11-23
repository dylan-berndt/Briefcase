from flask import Flask, jsonify, request, abort

import json
import os

import hashlib
import secrets

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

freeDomains = ["https://www.dafont.com", "https://fonts.google.com", "https://www.fontsquirrel.com"]
paidDomains = ["https://myfonts.com"]

# SQL? What's that?
with open("vectors.json", "r") as file:
    fontData = json.load(file)
    fontNames = np.array([key for key in fontData])
    fontVectors = torch.stack([torch.tensor(fontData[key]["vector"], dtype=torch.float32) for key in fontData], dim=0)
    fontPaid = np.array([fontData[key]["paid"] for key in fontData], dtype=bool)

app = Flask(__name__)


def hashPassword(password, salt=None, iterations=260000):
    if salt is None:
        salt = secrets.token_hex(16)
    hashData = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return salt, hashData.hex()


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

    vectors = fontVectors
    names = fontNames
    if not includePaid:
        vectors = vectors[~fontPaid]
        names = names[~fontPaid]

    scores = torch.cosine_similarity(vectors, output.unsqueeze(1))
    scores = torch.mean(scores, dim=1)
    top20 = torch.argsort(scores, descending=True)[:20]
    topScores = scores[top20]
    names = names[top20.numpy()]

    results = [{"name": names[i], "score": topScores[i], "file": fontData[names[i]]["file"], "url": fontData[names[i]]["url"]} for i in range(20)]

    return jsonify({"results": results}), 200


@app.route('/api/font/update', methods=['GET'])
def updateRegistry():
    checkPassword()
    
    # TODO: Run image model for each new font added and append information to vectors.json
    return jsonify("Successful"), 200


@app.route('/api/font/add', methods=['GET'])
def addFontToRegistry():
    checkPassword()

    name, url, file = request.args.get("name", ""), request.args.get("url", ""), request.args.get("file", "")
    if name == "" or url == "" or file == "":
        abort(400)

    urlGood = False
    fileGood = False
    paid = False
    for site in paidDomains + freeDomains:
        if url.startswith(site):
            urlGood = True
            if site in paidDomains:
                paid = True
        if file.startswith(site):
            fileGood = True
            if site in paidDomains:
                paid = True

    if not urlGood or not fileGood:
        abort(400)
    
    # TODO: Add fonts to some kind of registry to be ran later


@app.route('/api/font/change/', methods=['GET'])
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


