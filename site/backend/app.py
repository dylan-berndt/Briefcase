from flask import Flask, jsonify, request

import json
import os

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


app = Flask(__name__)


@app.route('/api/font/query/<query>', methods=['GET'])
def findFonts(query):
    pass




