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


def compressEmbeddings(embeddings, components=12, method="PCA", gpu=False):
    if method == "PCA":
        pca = PCA(n_components=components)

        values = np.stack(list(embeddings.values()), axis=0)
        transformed = pca.fit_transform(values)

    if method == "TSNE":
        pca = PCA(n_components=20)

        values = np.stack(list(embeddings.values()), axis=0)
        transformed = pca.fit_transform(values)

        if gpu:
            import torch
            import torchdr

            # torchdr only has squared-euclidean; on unit-norm rows that is 2x cosine distance, which is
            # equivalent here (perplexity calibration is scale-free). Sparse kNN affinities stand in for
            # sklearn's dense "exact" mode, which would need a 40k x 40k matrix on the GPU.
            transformed = transformed / np.maximum(np.linalg.norm(transformed, axis=1, keepdims=True), 1e-12)
            from torch.utils.checkpoint import checkpoint

            class ChunkedTSNE(torchdr.TSNE):
                # torchdr's repulsive term builds the full N x N distance matrix (and autograd keeps it);
                # at 40k points that is >6 GB. Same loss (log sum_ij 1/(1+d_ij), diagonal included), computed in
                # row chunks and recomputed in backward, so memory is O(chunk * N).
                def _compute_repulsive_loss(self, chunk=1024):
                    Z = self.embedding_
                    zNorm = (Z * Z).sum(1)

                    def part(rows, rowNorm):
                        d = (rowNorm[:, None] + zNorm[None, :] - 2 * rows @ Z.T).clamp_min(0)
                        return torch.logsumexp(-torch.log1p(d), dim=(0, 1))

                    parts = [checkpoint(part, Z[i:i + chunk], zNorm[i:i + chunk], use_reentrant=False)
                             for i in range(0, Z.shape[0], chunk)]
                    return torch.logsumexp(torch.stack(parts), dim=0)

            tsne = ChunkedTSNE(n_components=components, perplexity=30.0, lr="auto", init="pca", random_state=42, metric="sqeuclidean", device="cuda", sparsity=True, verbose=True)
            transformed = tsne.fit_transform(torch.from_numpy(transformed.astype(np.float32)))
            transformed = transformed.detach().cpu().numpy()
        else:
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