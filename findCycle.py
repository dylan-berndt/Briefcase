# Utility to find cycles inside the font embeddings
# Useful for generating the list of fonts to use on the front page of the site
# Using Google fonts for easy imports on the site
from utils import *
from sklearn.decomposition import PCA

latinCharacters = latin + [c.upper() for c in latin]


def generateEmbeddings(fontData, model):
    embeddings = {}

    paths = dict(zip([(fontData["names"][i], fontData["letters"][i]) for i in range(len(fontData["names"]))], fontData["paths"]))

    for name in np.unique(fontData["names"]):
        images = []
        for letter in latinCharacters:
            path = paths[(name, letter)]
            images.append(torch.tensor(loadImage(path), dtype=torch.float32))

        batch = torch.stack(images, dim=0).unsqueeze(-1)
        fontEmbeddings = nn.functional.normalize(model(batch), dim=-1).mean(dim=0)

        embeddings[name] = fontEmbeddings.numpy()

    return embeddings


def compressEmbeddings(embeddings, components=12):
    pca = PCA(n_components=components)

    values = np.stack(embeddings.values(), axis=0)
    transformed = pca.fit_transform(values)

    return dict(zip(embeddings.keys(), transformed))


def greedyPaths(embeddings, 
                initialPaths = 2000, nodes = 20, 
                distanceAlpha = 0.2, maxAlpha = 0.2, varianceAlpha = 0.2, outAlpha = 0.2, 
                replaceMutationRate = 0.05, flipMutationRate = 0.3,
                trainingSteps = 2000, topK = 300) -> list[list[str]]:
    paths = [random.sample(list(embeddings.keys()), nodes) for i in range(initialPaths)]
    names = list(embeddings.keys())

    left = np.arange(nodes)[:, None]
    right = np.arange(nodes)[None, :]

    mask = np.logical_or(np.logical_or(left - 1 == right, left + 1 == right), np.eye(nodes, dtype=bool))
    edgesOnly = np.logical_or(left - 1 == right, left + 1 == right)

    def scorePath(path):
        cycle = path + [path[0]]
        emb = np.stack([embeddings[name] for name in cycle], axis=0)
        distances = 1 - (emb[:, None] @ emb[None, :])

        maxDistanceScore = np.mean(distances[~mask])
        minEdgesScore = np.mean(distances[edgesOnly])
        edgeVarianceScore = np.std(distances[edgesOnly])
        maxValuesScore = np.mean(np.abs(emb))

        score = maxAlpha * maxDistanceScore - distanceAlpha * minEdgesScore + varianceAlpha * edgeVarianceScore + outAlpha * maxValuesScore
        return score
    
    def mutateReplace(path):
        if random.random() > replaceMutationRate:
            return path
        path = path.copy()
        available = [f for f in names if f not in path]
        if not available:
            return path
        
        i = random.randrange(nodes)
        path[i] = random.choice(available)
        return path
        
    def mutateFlip(path):
        if random.random() > flipMutationRate:
            return path
        path = path.copy()
        i, j = sorted(random.sample(range(nodes), 2))
        path[i: j + 1] = path[i: j + 1][::-1]
        return path
    
    def select(scores, paths, topK):
        scores = np.array(scores)
        # Temperature controls selection pressure; higher = more uniform sampling
        temperature = np.std(scores) + 1e-8
        logits = (scores - scores.mean()) / temperature
        probs = np.exp(logits) / np.exp(logits).sum()
        chosen = np.random.choice(len(paths), size=topK, replace=False, p=probs)
        return [scores[i] for i in chosen], [paths[i] for i in chosen]
    
    def reproduce(parents, targetSize):
        children = []
        while len(children) < targetSize:
            a, b = random.sample(parents, 2)
            i, j = sorted(random.sample(range(nodes), 2))
            child = [None] * nodes
            child[i: j + 1] = a[i: j + 1]
            fill = [x for x in b if x not in child]
            idx = 0
            for k in range(nodes):
                if child[k] is None:
                    child[k] = fill[idx]
                    idx += 1

            child = mutateFlip(mutateReplace(child))
            children.append(child)
        return children
    
    scores = [scorePath(p) for p in paths]

    for step in range(trainingSteps):
        scores, paths = select(scores, paths, topK)

        if step % 100 == 0:
            print(f"Step: {step:4d} | Best: {scores[0]:.4f} | Mean: {np.mean(scores):.4f}")

        paths = paths + reproduce(paths, initialPaths - topK)
        scores = scores + [scorePath(p) for p in paths[topK:]]

    ranked = sorted(zip(scores, paths), key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked[:topK]]


if __name__ == "__main__":
    model, config = ViT.load(os.path.join("checkpoints", "pretrain", "latest"))
    data = collectFontSetPaths("google", config.dataset.fontSize, "bitmaps")

    embeddings = generateEmbeddings(data, model)
    compressed = compressEmbeddings(embeddings)

    paths = greedyPaths(embeddings)

    for path in paths:
        print(" -> ".join(path))