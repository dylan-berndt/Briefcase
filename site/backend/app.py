from flask import Flask, jsonify, request, abort

import json
import os

from dotenv import load_dotenv
import hashlib
import secrets

# testDir = os.path.join("checkpoints", "finetune", "best")
# textModel = CLIPTextModel.from_pretrained(os.path.join(testDir, "text"))
# Description.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# imageModel, conf = UNet.load(testDir, name="image")
# conf.dataset.directory = "google"
# dataset = FontData(conf.dataset, training=False)

# textModel.eval()
# imageModel.eval()

# TODO: Load fonts and models, precompute vectors
# TODO: Store precomputed vectors

load_dotenv()
hashword = os.environ.get("ADMIN")

acceptableSites = ["https://www.dafont.com", "https://fonts.google.com", "https://www.fontsquirrel.com/"]


app = Flask(__name__)


def hashPassword(password, iterations=260000):
    salt = secrets.token_hex(16)
    hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return hash


def checkPassword():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        password = auth.split(" ", 1)[1].strip()
        hashed = hashPassword(password)
        if secrets.compare_digest(hashword, hashed):
            return
        abort(jsonify({"error": "Unauthorized"}), 401)

    abort(jsonify({"error": "Unauthorized"}), 401)


@app.route('/api/font/query/<query:str>', methods=['GET'])
def findFonts(query):
    pass


@app.route('/api/font/update', methods=['POST'])
def updateRegistry():
    checkPassword()
    
    # TODO: Run image model for each new font added


@app.route('/api/font/add/<name:str>/<url:str>/<file:str>', methods=['POST'])
def addFontToRegistry(name, url, file):
    checkPassword()
    
    # TODO: Add fonts to some kind of registry


@app.route('/api/font/change/<newPassword:str>', methods=['POST'])
def changeAdminPassword(newPassword):
    checkPassword()

    hashword = hashPassword(newPassword)




