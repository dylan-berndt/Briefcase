import torch
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.preprocessing import normalize
import numpy as np


def regressionTransform(z, y, clusters, decompositionRate, clusteringMetric):
    _, dimension = y.shape
    y = y.cpu().numpy()

    pca = PCA(dimension // decompositionRate)
    transformed = pca.fit_transform(y)

    clustering = AgglomerativeClustering(clusters, linkage="average", metric=clusteringMetric)
    clusters = clustering.fit_predict(normalize(transformed))

    return z, torch.tensor(clusters, dtype=torch.long)


# https://arxiv.org/abs/2106.09362
def codingRate(z, eps=1e-4):
    n, d = z.shape
    _, rate = torch.linalg.slogdet((torch.eye(d) + 1 / (n * eps) * z.transpose(0, 1) @ z))
    return 0.5 * rate


def transRate(z, y, clusters=48, decompositionRate=4, clusteringMetric="cosine", eps=1e-4):
    z, y = regressionTransform(z, y, clusters, decompositionRate=decompositionRate, clusteringMetric=clusteringMetric)

    z = z - torch.mean(z, axis=0, keepdim=True)
    rz = codingRate(z, eps)
    rzy = 0
    k = int(y.max() + 1)
    for i in range(k):
        rzy += codingRate(z[(y == i).flatten()], eps)
    return (rz - rzy / k).item()


# https://github.com/OpenGVLab/Multitask-Model-Selector
def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ y_.contiguous()))
    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
        # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if torch.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * torch.log(alpha) \
               + N / 2.0 * torch.log(beta) \
               - 0.5 * torch.sum(torch.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * torch.log(2 * np.pi)
    return evidence / N, alpha, beta, m


def truncated_svd(x):
    u, s, vh = torch.linalg.svd(x.transpose(0, 1) @ x)
    s = torch.sqrt(s)
    u_times_sigma = x @ vh.transpose(0, 1)
    k = torch.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh


def logME(z, y):
    fh = z
    f = z.transpose(0, 1)
    D, N = f.shape
    v, s, vh = torch.linalg.svd(f @ fh, full_matrices=True)

    alphas = []
    betas = []
    ms = []

    evidences = []
    dimensions = y.shape[1]
    for i in range(dimensions):
        y_ = y[:, i]
        evidence, alpha, beta, m = each_evidence(y_, f, fh, v, s, vh, N, D)
        evidences.append(evidence)
        alphas.append(alpha)
        betas.append(beta)
        ms.append(m)
    ms = torch.stack(ms)
    return torch.mean(evidences).item()

