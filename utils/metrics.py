import torch
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge
import numpy as np


def discretizeTransRate(z, y, numBins=10):
    yScalar = y.mean(axis=1)

    n = y.shape[0]
    idx = torch.argsort(yScalar)
    bins = torch.zeros(n, dtype=torch.long)

    binSize = n // numBins
    for c in range(numBins):
        start = c * binSize
        end = (c + 1) * binSize if c < numBins - 1 else n
        bins[idx[start:end]] = c

    return z, bins


def discretizeClustering(z, y, clusters, decompositionRate, clusteringMetric):
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
    _, rate = torch.linalg.slogdet((torch.eye(d) + 1 / (n * eps) * z.t() @ z))
    return 0.5 * rate


def transRate(z, y, fixedDimension=24, clusters=24, decompositionRate=4, clusteringMetric="euclidean", eps=1e-4, fixed=True):
    z, y = discretizeClustering(z, y, clusters, decompositionRate=decompositionRate, clusteringMetric=clusteringMetric)
    # z, y = discretizeTransRate(z, y)

    if fixed:
        pca = PCA(fixedDimension)
        transformed = pca.fit_transform(z.cpu().numpy())
        z = torch.tensor(transformed, dtype=torch.float32)

    z = z - torch.mean(z, dim=0, keepdim=True)
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
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m


def truncated_svd(x):
    u, s, vh = torch.linalg.svd(x.t() @ x)
    s = torch.sqrt(s)
    u_times_sigma = x @ vh.t()
    k = torch.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh


def logME(z, y):
    fh = z
    f = z.t()
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
    return torch.mean(torch.stack(evidences)).item()


# https://arxiv.org/pdf/2312.00656v2
def linearMSE(z, y):
    z = z.cpu().numpy()
    y = y.cpu().numpy()

    regressor = Ridge()
    regressor.fit(z, y)

    return -np.mean(np.power(y - regressor.predict(z), 2))


# https://arxiv.org/pdf/2110.06893
def regularizedCovariance(A, f):
    ef = (f.t() @ f) / f.shape[0]
    return (1 - A) * ef + (A * torch.eye(f.shape[1]))


def computeAlpha(f):
    n, d = f.shape
    ef = (f.t() @ f) / f.shape[0]

    # TODO: Speed up and verification
    term1 = 0
    for i in range(n):
        fi = f[i].unsqueeze(1)
        diff = fi @ fi.t() - ef
        term1 += torch.sum(diff * diff)
    term1 = term1 / (n ** 2)

    efc = ef - torch.trace(ef) / d * torch.eye(d)
    term2 = torch.sum(efc * efc)

    a = (term1 / term2).item()

    return min(a, 1)


def hAlphaScore(z, y, fixedDimension=24, clusters=24, decompositionRate=4, clusteringMetric="euclidean", eps=1e-4):
    z, y = discretizeClustering(z, y, clusters, decompositionRate=decompositionRate, clusteringMetric=clusteringMetric)
    z += eps

    pca = PCA(fixedDimension)
    transformed = pca.fit_transform(z.cpu().numpy())
    # TODO: Double-check normalization
    z = normalize(transformed)
    z = torch.tensor(z, dtype=torch.float32)

    n, d = z.shape
    C = int(torch.amax(y) + 1)

    unique, counts = torch.unique(y, sorted=True, return_counts=True)
    R = torch.zeros(d, C)
    for c, cls in enumerate(unique):
        # TODO: Make sure this is correct also
        mask = y == cls
        mf = z[mask].mean(0)
        R[:, c] = torch.sqrt(counts[c]) * mf

    A = computeAlpha(z)

    if n < d:
        s = 1
        w = n * A * s * np.eye(n) + (1 - A) * (z @ z.t())
        g = z @ R
        # TODO: Double check the process here, maybe split for legibility
        hAlpha = ((1 - A) / (n * s * A)) @ (torch.norm(R) - ((1 - A) * (g.t() @ (torch.linalg.inv(w) @ g))))
        return hAlpha.item()
    
    efa = regularizedCovariance(A, z)
    hAlpha = ((1 - A) / n) * torch.trace((torch.linalg.inv(efa) @ R) @ R.t())
    return hAlpha.item()


if __name__ == "__main__":
    samples = 2048
    dimensions = [64, 128, 256, 512]
    tests = 100

    for dim1 in dimensions:
        for dim2 in dimensions:
            t1, t2, l, m, h = 0, 0, 0, 0, 0
            for i in range(tests):
                x1 = torch.randn([samples, dim1])
                y1 = torch.randn([samples, dim2])

                t1 += transRate(x1, y1) / tests
                t2 += transRate(x1, y1, fixed=False) / tests
                l += logME(x1, y1) / tests
                m += linearMSE(x1, y1) / tests
                h += hAlphaScore(x1, y1) / tests

            # print(f"Dimensions: {dimension} || TransRate: {t:.2f} | LogME: {l:.2f} | LinMSE: {m:.2f} | H-Alpha: {h:.2f}")

            print(f"Dimensions: {dim1}, {dim2} || TransRate (Fixed): {t1:.2f} | TransRate (Standard): {t2:.2f} | LogME: {l:.4f} | LinMSE: {m:.2f} | H-Alpha: {h:.2f}")
