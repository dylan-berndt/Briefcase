import torch
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.preprocessing import normalize


def codingRate(z, eps=1e-4):
    n, d = z.shape
    _, rate = torch.linalg.slogdet((torch.eye(d) + 1 / (n * eps) * z.transpose() @ z))
    return 0.5 * rate


def transRate(z, y, eps=1e-4):
    z = z - torch.mean(z, axis=0, keepdim=True)
    rz = codingRate(z, eps)
    rzy = 0
    k = int(y.max() + 1)
    for i in range(k):
        rzy += codingRate(z[(y == i).flatten()], eps)
    return rz - rzy / k


def logME(z, y):
    pass


def regressionTransform(z, y, clusters, decompositionRate=4, clusteringMetric="cosine"):
    _, dimension = y.shape
    y = y.numpy()

    pca = PCA(dimension // decompositionRate)
    transformed = pca.fit_transform(y)

    clustering = AgglomerativeClustering(clusters, linkage="average", metric=clusteringMetric)
    clusters = clustering.fit_predict(normalize(transformed))

    return z, torch.tensor(y, dtype=torch.long)
