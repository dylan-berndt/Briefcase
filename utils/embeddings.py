import torch
import torch.nn as nn
import json
import os
import numpy as np
from .pretraining import latin, loadImage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

latinCharacters = [chr(c) for c in latin]

@torch.no_grad()
def generateEmbeddings(fontData, model, name="google"):
    embeddings = {}

    os.makedirs("embeddings", exist_ok=True)
    if os.path.exists(os.path.join("embeddings", f"{name}.json")):
        with open(os.path.join("embeddings", f"{name}.json"), "r") as file:
            embeddings = json.load(file)
            embeddings = {key: np.array(value) for key, value in embeddings.items()}
            return embeddings

    paths = dict(zip([(fontData["names"][i], fontData["letters"][i]) for i in range(len(fontData["names"]))], fontData["paths"]))

    names = np.unique(fontData["names"])

    for n, name in enumerate(names):
        images = []
        broken = False
        for letter in latinCharacters:
            if (name, letter) not in paths:
                broken = True
                break
            path = paths[(name, letter)]
            _, image = loadImage(path)
            images.append(torch.tensor(image, dtype=torch.float32))

        if broken:
            continue

        batch = torch.stack(images, dim=0).unsqueeze(-1)
        _, embedding = model(batch)
        fontEmbeddings = nn.functional.normalize(embedding, dim=-1).mean(dim=0)

        embeddings[name] = fontEmbeddings.numpy()

        print(f"\r{n + 1}/{len(names)} Embeddings extracted", end="")

    print()

    with open(os.path.join("embeddings", f"{name}.json"), "w+") as file:
        serialized = {key: value.tolist() for key, value in embeddings.items()}
        json.dump(serialized, file)

    return embeddings


def compressEmbeddings(embeddings, components=12, method="PCA"):
    if method == "PCA":
        pca = PCA(n_components=components)

        values = np.stack(list(embeddings.values()), axis=0)
        transformed = pca.fit_transform(values)

    if method == "TSNE":
        pca = PCA(n_components=20)

        values = np.stack(list(embeddings.values()), axis=0)
        transformed = pca.fit_transform(values)

        tsne = TSNE(n_components=components, perplexity=30.0, learning_rate='auto', init='pca', random_state=42, method="exact")
        transformed = tsne.fit_transform(transformed)

    return dict(zip(embeddings.keys(), transformed))