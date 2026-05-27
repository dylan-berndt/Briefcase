# Utility to find cycles inside the font embeddings
# Useful for generating the list of fonts to use on the front page of the site
# Using Google fonts for easy imports on the site
from utils import *


@torch.no_grad()
def greedyPaths(embeddings, 
                initialPaths = 2000, nodes = 40, 
                distanceAlpha = 2, maxAlpha = 2.0, varianceAlpha = 2.0, outAlpha = 0.7, 
                replaceMutationRate = 0.1, flipMutationRate = 0.3,
                trainingSteps = 2000, topK = 400) -> list[list[str]]:
    paths = [random.sample(list(embeddings.keys()), nodes) for i in range(initialPaths)]
    names = list(embeddings.keys())

    left = np.arange(nodes + 1)[:, None]
    right = np.arange(nodes + 1)[None, :]

    mask = np.logical_or(np.logical_or(left - 1 == right, left + 1 == right), np.eye(nodes + 1, dtype=bool))
    edgesOnly = np.logical_or(left - 1 == right, left + 1 == right)

    idx = np.arange(nodes + 1)
    hopMatrix = np.minimum(np.abs(idx[:, None] - idx[None, :]), nodes - np.abs(idx[:, None] - idx[None, :]))

    nonEdgeMask = ~mask
    weights = hopMatrix[nonEdgeMask].astype(float)
    weights /= weights.sum()

    def scorePath(path):
        cycle = path + [path[0]]
        emb = np.stack([embeddings[name] for name in cycle], axis=0)
        distances = 1 - (emb @ emb.T)
    
        maxDistanceScore = np.sum(distances[nonEdgeMask] * weights)
        minEdgesScore = np.mean(distances[edgesOnly])
        edgeVarianceScore = np.std(distances[edgesOnly])
        maxValuesScore = np.mean(np.abs(emb))

        # print(minEdgesScore)

        score = maxAlpha * maxDistanceScore - distanceAlpha * minEdgesScore - varianceAlpha * edgeVarianceScore + outAlpha * maxValuesScore
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
    
    def reproduce(parents, targetSize, minDiversity=0.6):
        children = []
        while len(children) < targetSize:
            a, b = random.sample(parents, 2)
            if len(set(a) & set(b)) / nodes > (1 - minDiversity):
                continue
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

        print(f"\rStep: {step:4d} | Best: {max(scores):.4f} | Mean: {np.mean(scores):.4f}", end="")

        paths = paths + reproduce(paths, initialPaths - topK)
        scores = [scorePath(p) for p in paths]

        # scores = nichedScores(scores, paths)

    print()

    ranked = sorted(zip(scores, paths), key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked[:topK]]


if __name__ == "__main__":
    model, config = ViT.load(os.path.join("checkpoints", "pretrain", "latest"))
    data = collectFontSetPaths("google", config.dataset.fontSize, "bitmaps")

    embeddings = generateEmbeddings(data, model)
    compressed = compressEmbeddings(embeddings)

    paths = greedyPaths(embeddings)

    for path in paths[:10]:
        print(" -> ".join(path))