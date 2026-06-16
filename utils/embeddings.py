import torch
import torch.nn as nn
import json
import os
import numpy as np
from .pretraining import latin, loadImage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
from umap import UMAP

latinCharacters = [chr(c) for c in latin]

@torch.no_grad()
def generateEmbeddings(fontData, model, fileName="google"):
    embeddings = {}

    os.makedirs("embeddings", exist_ok=True)
    if os.path.exists(os.path.join("embeddings", f"{fileName}.json")):
        with open(os.path.join("embeddings", f"{fileName}.json"), "r") as file:
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

        batch = torch.stack(images, dim=0).unsqueeze(-1).to(next(model.parameters()).device)
        if type(model).__name__ == "ViT":
            _, embedding = model(batch)
        else:
            embedding = model(batch)
        # Old functionality that produced the flower
        fontEmbeddings = nn.functional.normalize(embedding, dim=-1).mean(dim=0)
        # fontEmbeddings = embedding.mean(dim=0).squeeze()

        embeddings[name] = fontEmbeddings.cpu().numpy()

        print(f"\r{n + 1}/{len(names)} Embeddings extracted", end="")

    print()

    with open(os.path.join("embeddings", f"{fileName}.json"), "w+") as file:
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

        tsne = TSNE(n_components=components, perplexity=30.0, learning_rate='auto', init='pca', random_state=42, method="exact", metric="cosine", n_jobs=6)
        transformed = tsne.fit_transform(transformed)

    if method == "UMAP":
        pca = PCA(n_components=80)

        values = np.stack(list(embeddings.values()), axis=0)
        transformed = pca.fit_transform(values)

        graph = kneighbors_graph(
            values,
            n_neighbors=10,
            metric="cosine",
            mode="connectivity",
            include_self=False
        )

        # connected components
        _, labels = connected_components(graph)

        # component sizes
        counts = np.bincount(labels)

        # keep only large components
        keep_components = np.where(counts > 100)[0]

        mask = np.isin(labels, keep_components)

        values = values[mask]
        keys = np.array(list(embeddings.keys()))[mask]

        umap = UMAP(n_components=components, n_neighbors=200, min_dist=0.4, random_state=42, metric="cosine", repulsion_strength=0.4)
        transformed = umap.fit_transform(values)
        return dict(zip(keys, transformed))

    return dict(zip(embeddings.keys(), transformed))