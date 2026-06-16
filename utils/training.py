import copy

import torch
import torch.nn as nn


# https://github.com/kreasof-ai/sigreg
def sigreg_weak_loss(x, sketch_dim=64):
    """
    Forces Covariance(x) ~ Identity.
    Matches the 2nd Moment (Spherical Cloud).
    """
    N, C = x.size()
    # 1. Sketching (Optional for C=512, but good for consistency)
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
        x = x @ S.T  # [N, sketch_dim]
    else:
        sketch_dim = C

    # 2. Centering & Covariance
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)

    # 3. Target Identity
    target = torch.eye(sketch_dim, device=x.device)

    # 4. Off-diagonal suppression + Diagonal maintenance
    return torch.norm(cov - target, p='fro')


class MomentumEncoder(nn.Module):
    """
    EMA copy of an encoder, used to produce the keys stored in a MoCoQueue.
    Kept frozen and in eval mode so keys stay consistent across batches.
    """
    def __init__(self, model: nn.Module, momentum: float = 0.999):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.model.eval()
        self.momentum = momentum

    @torch.no_grad()
    def update(self, model: nn.Module):
        for param, emaParam in zip(model.parameters(), self.model.parameters()):
            emaParam.mul_(self.momentum).add_(param.detach(), alpha=1.0 - self.momentum)
        for buffer, emaBuffer in zip(model.buffers(), self.model.buffers()):
            emaBuffer.copy_(buffer)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class MoCoQueue:
    def __init__(self, dim: int, size: int = 8192):
        self.queue = nn.functional.normalize(torch.randn(size, dim), dim=-1)
        self.text_queue = nn.functional.normalize(torch.randn(size, dim), dim=-1)
        self.families = torch.full((size,), fill_value=-1, dtype=torch.long)
        self.ptr = 0
        self.size = size

    def get(self):
        valid = self.families != -1
        if valid.sum() == 0:
            empty = torch.zeros(0, self.queue.shape[1])
            return empty, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
        return self.queue[valid].clone(), self.text_queue[valid].clone(), self.families[valid].clone()

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, text_embeddings: torch.Tensor, families: torch.Tensor):
        device = embeddings.device
        self.queue = self.queue.to(device)
        self.text_queue = self.text_queue.to(device)
        n = embeddings.shape[0]
        slots = torch.arange(self.ptr, self.ptr + n) % self.size
        self.queue[slots.to(device)] = nn.functional.normalize(embeddings.detach(), dim=-1)
        self.text_queue[slots.to(device)] = nn.functional.normalize(text_embeddings.detach(), dim=-1)
        self.families[slots] = families.detach().cpu()
        self.ptr = (self.ptr + n) % self.size


def buildFalseNegativeMask(descriptions, fontNum, threshold=0.35, chunk=2048):
    """
    TF-IDF cosine similarity between fonts' query texts -> [N, N] bool matrix where
    True means two fonts read as the same style and shouldn't be negatives for each
    other. Rows/cols are ordered by fontNum, so it can be indexed with family IDs.
    Threshold 0.35 chosen from the midQueries.json similarity distribution: pairs
    above it are visually interchangeable descriptions, ~1 similar font per font.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    names = sorted(fontNum, key=fontNum.get)
    docs = [" ".join(descriptions[name]) if isinstance(descriptions[name], list)
            else str(descriptions[name]) for name in names]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", max_features=50000)
    tfidf = vectorizer.fit_transform(docs)  # rows are l2-normalized, so tfidf @ tfidf.T is cosine

    n = len(names)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for start in range(0, n, chunk):
        block = (tfidf[start:start + chunk] @ tfidf.T).toarray()
        mask[start:start + chunk] = torch.from_numpy(block >= threshold)
    mask.fill_diagonal_(True)

    print(f"False negative mask: {n} fonts, avg {((mask.sum() - n) / n).item():.2f} similar fonts each")
    return mask


class EmbeddingLoss(nn.Module):
    def __init__(self, temperature=0.04, alpha=0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha       = alpha

    def infoNCE(self, queries, keys, queryFamilies, keyFamilies, simMask=None):
        """
        queries: [B, D]  — image or text embeddings from current batch
        keys:    [B+Q, D] — current batch + queue embeddings
        queryFamilies: [B]
        keyFamilies:   [B+Q]
        simMask: [N, N] bool — True where two fonts are too similar to be negatives
        """
        queries = nn.functional.normalize(queries, dim=-1)
        keys = nn.functional.normalize(keys, dim=-1)

        logits = (queries @ keys.T) / self.temperature  # [B, B+Q]

        # Positive mask: same family OR diagonal (exact pair)
        familyMatch = (queryFamilies.unsqueeze(1) == keyFamilies.unsqueeze(0))  # [B, B+Q]
        # Diagonal of the batch portion is always the exact positive
        familyMatch[:, :queryFamilies.shape[0]].fill_diagonal_(True)

        if simMask is not None:
            similar = simMask[queryFamilies.cpu()][:, keyFamilies.cpu()].to(logits.device)
            # Similar-but-different fonts are false negatives: drop them from the softmax.
            # Finite fill instead of -inf so the 0-target entries can't produce NaN.
            logits = logits.masked_fill(similar & ~familyMatch, -1e4)

        targets = familyMatch.float()
        targets = targets / targets.sum(dim=1, keepdim=True)

        logSoftmax = torch.log_softmax(logits, dim=1)
        return -(targets * logSoftmax).sum(dim=1).mean()

    def forward(self, x, y, families=None, queue: MoCoQueue = None, enqueue: bool = True, simMask=None):
        B = x.shape[0]
        device = x.device

        if families is not None and queue is not None:
            queueImg, queueTxt, queueFam = queue.get()
            queueImg = queueImg.to(device)
            queueTxt = queueTxt.to(device)
            queueFam = queueFam.to(device)

            allFamilies = torch.cat([families, queueFam], dim=0)

            # text queries → [current image batch ‖ image queue] keys
            imageKeys = torch.cat([x, queueImg], dim=0)
            loss1 = self.infoNCE(y, imageKeys, families, allFamilies, simMask=simMask)

            # image queries → [current text batch ‖ text queue] keys
            textKeys = torch.cat([y, queueTxt], dim=0)
            loss2 = self.infoNCE(x, textKeys, families, allFamilies, simMask=simMask)

            if enqueue:
                queue.enqueue(x, y, families)

        elif families is not None:
            xNorm, yNorm = nn.functional.normalize(x, dim=-1), nn.functional.normalize(y, dim=-1)
            logits = (xNorm @ yNorm.T) / self.temperature
            familyMatch  = (families.unsqueeze(0) == families.unsqueeze(1))
            familyMatch.fill_diagonal_(True)
            if simMask is not None:
                similar = simMask[families.cpu()][:, families.cpu()].to(device)
                # symmetric, so masking once covers the transposed direction too
                logits = logits.masked_fill(similar & ~familyMatch, -1e4)
            targets = familyMatch.float()
            targets = targets / targets.sum(dim=1, keepdim=True)
            logSoftmax = torch.log_softmax(logits, dim=1)
            logSoftmax2 = torch.log_softmax(logits.T, dim=1)
            loss1 = -(targets * logSoftmax).sum(dim=1).mean()
            loss2 = -(targets.T * logSoftmax2).sum(dim=1).mean()

        else:
            xNorm, yNorm = nn.functional.normalize(x, dim=-1), nn.functional.normalize(y, dim=-1)
            logits = (xNorm @ yNorm.T) / self.temperature
            labels = torch.arange(B, device=device)
            loss1 = nn.functional.cross_entropy(logits,   labels)
            loss2 = nn.functional.cross_entropy(logits.T, labels)

        infoLoss = 0.5 * (loss1 + loss2)
        sigLoss  = 0.5 * (sigreg_weak_loss(x) + sigreg_weak_loss(y))
        return {"total": infoLoss + self.alpha * sigLoss, "info": infoLoss, "sig": sigLoss}


# class Perplexity(nn.Module):
#     def __init__(self, loss):
#         super().__init__()
#         self.loss = loss

#     def forward(self, y1, y2, names, queue):
#         log = self.loss(y1, y2, names, queue)["info"]
#         return torch.exp(log)
    

def recallAtK(x, y, families=None, k=10, queue: MoCoQueue = None, simMask=None):
    B = x.shape[0]
    device = x.device
    xNorm = nn.functional.normalize(x, dim=-1)
    yNorm = nn.functional.normalize(y, dim=-1)

    if queue is not None:
        queueEmb, _, queueFam = queue.get()
        queueEmb = nn.functional.normalize(queueEmb.to(device), dim=-1)
        queueFam = queueFam.to(device)
        keys = torch.cat([xNorm, queueEmb], dim=0)
        allFamilies = torch.cat([families, queueFam], dim=0)
    else:
        keys = xNorm
        allFamilies = families

    logits = yNorm @ keys.T  # [B, B+Q]

    if allFamilies is not None:
        positive = (families.unsqueeze(1) == allFamilies.unsqueeze(0))  # [B, B+Q]
    else:
        positive = torch.zeros(B, keys.shape[0], dtype=torch.bool, device=device)
        positive[:, :B].fill_diagonal_(True)

    if simMask is not None and allFamilies is not None:
        similar = simMask[families.cpu()][:, allFamilies.cpu()].to(device)
        # similar fonts aren't fair distractors — push them out of the ranking
        logits = logits.masked_fill(similar & ~positive, -1e4)

    topK = logits.topk(k, dim=1).indices
    hits = positive.gather(1, topK).any(dim=1).float()
    return hits.mean().item()


def medianRank(x, y, families=None, queue: MoCoQueue = None, simMask=None):
    """
    Median rank (1-indexed) of the best-ranked correct image key for each text query,
    over [current image batch ‖ image queue] keys. 1 is perfect; lower is better.
    """
    B = x.shape[0]
    device = x.device
    xNorm = nn.functional.normalize(x, dim=-1)
    yNorm = nn.functional.normalize(y, dim=-1)

    if queue is not None:
        queueEmb, _, queueFam = queue.get()
        queueEmb = nn.functional.normalize(queueEmb.to(device), dim=-1)
        queueFam = queueFam.to(device)
        keys = torch.cat([xNorm, queueEmb], dim=0)
        allFamilies = torch.cat([families, queueFam], dim=0)
    else:
        keys = xNorm
        allFamilies = families

    logits = yNorm @ keys.T  # [B, B+Q]

    if allFamilies is not None:
        positive = (families.unsqueeze(1) == allFamilies.unsqueeze(0))  # [B, B+Q]
    else:
        positive = torch.zeros(B, keys.shape[0], dtype=torch.bool, device=device)
        positive[:, :B].fill_diagonal_(True)

    if simMask is not None and allFamilies is not None:
        similar = simMask[families.cpu()][:, allFamilies.cpu()].to(device)
        # similar fonts aren't fair distractors — push them out of the ranking
        logits = logits.masked_fill(similar & ~positive, -1e4)

    order = logits.argsort(dim=1, descending=True)       # [B, B+Q]
    positiveSorted = positive.gather(1, order)
    firstHit = positiveSorted.float().argmax(dim=1) + 1  # rank of first positive per query
    return firstHit.float().median().item()