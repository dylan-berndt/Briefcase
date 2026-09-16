"""
Search managers that bundle the loose model / embedding / result-handling logic
that used to live in the top-level ``search.py`` script.

``FontSearch`` is the base class. It owns the models, the font glyph dataset,
the cached per-font embeddings, and the ranked results, exposing a single
``search(query, k)`` entry point. It reproduces the original CLIP image/text
embedding search as-is.

``TagSearch`` is a subclass that searches with the multi-label tagging models in
``checkpoints/retrieval`` (trained by ``retrieval.py`` / ``utils/querying.py``).
Each font is encoded as a probability over the checkpoint's tag vocabulary, and
a free-text query is mapped onto that vocabulary with spaCy before ranking.

``GridFeedbackSearch`` implements the "Interactive Grid-Feedback Retrieval" method:
no text query at all, just repeated rounds of picking (or skipping) fonts out of
a grid, refit as an L2-logistic-regression belief over a whitened PCA subspace of
raw (pre-projection) backbone features. See the design doc for the derivation.
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import math

from .config import Config
from .vit import ViT, ViTEmbedder
from .querying import CombinedQueryData, CLIPTextEmbedder
from .embeddings import generateEmbeddings, latinCharacters
from .pretraining import device, loadImage
from .loaders.description import Description

__all__ = ["FontSearch", "TagSearch", "ClusteringSearch", "MeanderSearch", "WalkSearch",
           "GridFeedbackSearch"]


DEFAULT_BACKBONE = os.path.join("checkpoints", "pretrain", "latest")
DEFAULT_FINETUNE = os.path.join("checkpoints", "finetune", "2026-06-07 17-04",
                                "ViT openai-clip-vit-base-patch32")
CLIP_NAME = "openai/clip-vit-base-patch32"

# Characters drawn for each result row (mirrors the old search.py UI).
DISPLAY_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz"

# Words that flip the next content term to a negative weight ("not fun", "less bouncy").
NEGATIONS = {"not", "no", "non", "without", "less", "least", "never", "anti", "un"}


class FontSearch:
    """
    Manages the models, font embeddings and search results for a single search
    backend. Subclasses customise how fonts are encoded (``loadModels`` /
    ``embedFonts``) and how a query is scored (``encodeQuery``); everything else
    -- the dataset, the name->glyph path map, top-k ranking and glyph rendering
    -- is shared.

    By default it reproduces the CLIP image/text embedding search: fonts are
    embedded with a finetuned ``ViTEmbedder`` and queries with ``CLIPTextEmbedder``.
    """

    def __init__(self, checkpoint=DEFAULT_FINETUNE, backbone=DEFAULT_BACKBONE,
                 dataset=None, embeddingName="allText", device=device):
        self.device = device
        self.checkpoint = checkpoint
        self.backbone = backbone
        self.embeddingName = embeddingName

        self.loadModels()
        self.dataset = dataset if dataset is not None else self.buildDataset()
        self.fontPathMap = self._buildPathMap()
        self.embeddings = self.embedFonts()

        # Result state, updated by search().
        self.query = ""
        self.rankings = {}
        self.results = []

    # ------------------------------------------------------------------ models
    def loadModels(self):
        """Load and stash the image / text models. Sets ``self.datasetConfig``."""
        _, backboneConf = ViT.load(self.backbone)
        self.imageModel, conf = ViTEmbedder.load(self.checkpoint,
                                                 model=ViT(backboneConf.model),
                                                 name="image")
        self.config = conf
        self.datasetConfig = conf.dataset

        self.textModel = CLIPTextEmbedder(CLIP_NAME, conf.model.embedDim)
        self.textModel.load_state_dict(
            torch.load(os.path.join(self.checkpoint, "text.pt"), map_location=self.device))
        Description.tokenizer = AutoTokenizer.from_pretrained(CLIP_NAME)

        self.imageModel.eval().to(self.device)
        self.textModel.eval().to(self.device)

    def buildDataset(self):
        return CombinedQueryData(self.datasetConfig, training=False)

    # -------------------------------------------------------------- embeddings
    def embedFonts(self):
        """Per-font embedding matrix, cached under ``embeddings/<name>.json``."""
        return generateEmbeddings(
            {"names": self.dataset.names,
             "paths": self.dataset.paths,
             "letters": self.dataset.letters},
            model=self.imageModel,
            fileName=self.embeddingName,
        )

    # ------------------------------------------------------------------- query
    @torch.no_grad()
    def encodeQuery(self, query):
        """Score every font against ``query``. Returns ``{name: score}``."""
        textData = Description.tokenizer([query], padding=False, return_tensors="pt")
        textData = {k: v.to(self.device) for k, v in textData.items() if k != "token_type_ids"}
        embeddedText = self.textModel(textData).cpu()

        keys = list(self.embeddings.keys())
        matrix = torch.tensor(np.stack([self.embeddings[k] for k in keys]), dtype=torch.float32)
        scores = matrix @ nn.functional.normalize(embeddedText, dim=-1).t()
        return {keys[i]: scores[i].item() for i in range(len(keys))}

    # ----------------------------------------------------------------- results
    def search(self, query, k=100):
        """Run a query, remember the result, and return the top-k ``(name, score)``."""
        self.query = query
        self.rankings = self.encodeQuery(query)
        self.results = self.topK(self.rankings, k)
        return self.results

    @staticmethod
    def topK(scores, k=5):
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]

    def glyphStrip(self, name, characters=DISPLAY_CHARACTERS):
        """
        Render a font's glyphs into one white-on-transparent RGBA array, laid out
        left to right. Returns ``None`` if the font has no usable glyphs. Callers
        (e.g. a pygame UI) can blit this directly.
        """
        letterMap = self.fontPathMap.get(name)
        if not letterMap:
            return None

        tiles = []
        for char in characters:
            path = letterMap.get(char)
            if path is None:
                continue
            _, arr = loadImage(path)
            if arr is None:
                continue
            arr = np.squeeze(arr)
            gray = (arr * 255).astype(np.uint8)
            h, w = gray.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = 255
            rgba[..., 3] = gray
            tiles.append(rgba)

        if not tiles:
            return None

        maxH = max(tile.shape[0] for tile in tiles)
        padded = [np.pad(tile, ((0, maxH - tile.shape[0]), (0, 0), (0, 0))) for tile in tiles]
        return np.concatenate(padded, axis=1)

    # ------------------------------------------------------------------ helpers
    def _buildPathMap(self):
        pathMap = {}
        for name, letter, path in zip(self.dataset.names, self.dataset.letters, self.dataset.paths):
            pathMap.setdefault(name, {})[letter] = path
        return pathMap


class TagSearch(FontSearch):
    """
    NLP search over the multi-label tag models in ``checkpoints/retrieval``.

    The retrieval head emits one logit per tag in the checkpoint's ``vocab.json``;
    a font is encoded as the mean sigmoid probability of each tag across its latin
    glyphs. A free-text query is parsed with spaCy and mapped onto the vocabulary
    (lemma matches weigh 1.0, substring matches 0.5); fonts are then ranked by the
    dot product of their tag probabilities with those query weights.
    """

    def __init__(self, checkpoint=None, backbone=DEFAULT_BACKBONE, dataset=None,
                 experiment="ViT tags", spacyModel="en_core_web_lg",
                 synonymThreshold=0.5, synonymScale=1.0, device=device):
        if checkpoint is None:
            checkpoint = os.path.join("checkpoints", "retrieval", "latest")

        self.experiment = experiment
        self.spacyModel = spacyModel
        # Synonym matching needs word vectors (en_core_web_lg/md); a query term is
        # matched to vocab tags whose vector cosine >= synonymThreshold, weighted by
        # similarity * synonymScale (kept below the 1.0 of an exact lemma match).
        self.synonymThreshold = synonymThreshold
        self.synonymScale = synonymScale

        # Keep the tag-probability cache separate from the CLIP embedding cache,
        # and distinct per checkpoint so a re-trained model doesn't reuse stale vectors.
        stamp = os.path.basename(os.path.normpath(checkpoint))
        embeddingName = f"tags-{stamp}-{experiment}".replace(" ", "_")

        super().__init__(checkpoint=checkpoint, backbone=backbone, dataset=dataset,
                         embeddingName=embeddingName, device=device)

    # ------------------------------------------------------------------ models
    def loadModels(self):
        self.modelDir = os.path.join(self.checkpoint, self.experiment)

        vit, imageConf = ViT.load(self.backbone)
        self.datasetConfig = imageConf.dataset
        self.config = Config().load(os.path.join(self.modelDir, "config.json"))

        # The head's output width is the tag vocabulary size (see retrieval.py).
        self.model = ViTEmbedder(vit, self.config.numTags)
        head = torch.load(os.path.join(self.modelDir, "head.pt"), map_location=self.device)
        self.model.head.load_state_dict(head)
        self.model.eval().to(self.device)

        with open(os.path.join(self.modelDir, "vocab.json"), "r") as file:
            self.vocab = json.load(file)

        self._loadNLP()

    # -------------------------------------------------------------- embeddings
    @torch.no_grad()
    def embedFonts(self):
        """Mean per-tag probability for every font, cached like generateEmbeddings."""
        os.makedirs("embeddings", exist_ok=True)
        path = os.path.join("embeddings", f"{self.embeddingName}.json")
        if os.path.exists(path):
            with open(path, "r") as file:
                return {key: np.array(value) for key, value in json.load(file).items()}

        lookup = {(self.dataset.names[i], self.dataset.letters[i]): self.dataset.paths[i]
                  for i in range(len(self.dataset.names))}
        names = np.unique(self.dataset.names)

        embeddings = {}
        for n, name in enumerate(names):
            images = []
            broken = False
            for letter in latinCharacters:
                if (name, letter) not in lookup:
                    broken = True
                    break
                _, image = loadImage(lookup[(name, letter)])
                if image is None:
                    broken = True
                    break
                images.append(torch.tensor(image, dtype=torch.float32))

            if broken:
                continue

            batch = torch.stack(images, dim=0).unsqueeze(-1).to(self.device)
            probs = torch.sigmoid(self.model(batch)).mean(dim=0)
            embeddings[name] = probs.cpu().numpy()

            print(f"\r{n + 1}/{len(names)} tag vectors extracted", end="")

        print()
        with open(path, "w+") as file:
            json.dump({key: value.tolist() for key, value in embeddings.items()}, file)
        return embeddings

    # ------------------------------------------------------------------- query
    def encodeQuery(self, query):
        weights = self.queryWeights(query)

        keys = list(self.embeddings.keys())
        matrix = np.stack([self.embeddings[k] for k in keys], axis=0)  # [fonts, tags]
        scores = matrix @ weights
        return {keys[i]: float(scores[i]) for i in range(len(keys))}

    def queryWeights(self, query):
        """
        Map a free-text query onto a signed weight per vocabulary tag.

        Each content term is matched to tags by exact lemma (1.0), substring (0.5)
        and -- when word vectors are available -- vector-similar synonyms. A
        negation cue ("not", "less", ...) flips the following term to a negative
        weight. Positive and negative evidence are combined per tag and cancel, so
        "not fun" subtracts the "fun" tags from any fonts that have them.

        Also logs how many query terms found a vocabulary match, so a weak query
        can be diagnosed as out-of-vocabulary words vs. poor model predictions.
        """
        pos = np.zeros(len(self.vocab), dtype=np.float32)
        neg = np.zeros(len(self.vocab), dtype=np.float32)

        matched, unmatched = 0, []
        negate = False
        for token in self.nlp(query.lower()):
            if token.is_punct or token.is_space:
                negate = False  # punctuation ends a negation's scope
                continue
            # Negation cues are usually stopwords, so handle them before the filter.
            if token.lower_ in NEGATIONS or token.dep_ == "neg" or token.lower_ == "n't":
                negate = True
                continue
            if token.is_stop:
                continue

            hits = self._termMatches(token)
            if hits:
                matched += 1
                target = neg if negate else pos
                for i, magnitude in hits.items():
                    target[i] = max(target[i], magnitude)
            else:
                unmatched.append(token.text)
            negate = False

        total = matched + len(unmatched)
        print(f"Query '{query}': matched {matched}/{total} terms to the vocabulary"
              + (f" (no match: {', '.join(unmatched)})" if unmatched else ""))

        return pos - neg

    def _termMatches(self, token):
        """
        Return ``{tagIndex: magnitude}`` of vocabulary tags a single term hits.

        Direct matches (exact lemma 1.0, substring 0.5) take priority; vector-based
        synonyms are only used as a fallback when the term has no direct match, so an
        exact hit isn't diluted by loosely related tags.
        """
        matches = {}
        lemma, text = token.lemma_, token.text

        for i in self.lemmaToTags.get(lemma, []):
            matches[i] = max(matches.get(i, 0.0), 1.0)

        for i, tag in enumerate(self.vocab):
            tagLower = tag.lower()
            if (text and text in tagLower) or (lemma and lemma in tagLower):
                matches[i] = max(matches.get(i, 0.0), 0.5)

        if matches:
            return matches

        if self.tagVectors is not None and token.has_vector and token.vector_norm:
            sims = self.tagVectors @ (token.vector / token.vector_norm)
            for i in np.where(sims >= self.synonymThreshold)[0]:
                matches[int(i)] = float(sims[i]) * self.synonymScale

        return matches

    def tagsFor(self, name, k=10):
        """Top-k predicted tags for a font: ``[(tag, probability), ...]``."""
        probs = self.embeddings.get(name)
        if probs is None:
            return []
        order = np.argsort(probs)[::-1][:k]
        return [(self.vocab[i], float(probs[i])) for i in order]

    # ------------------------------------------------------------------ helpers
    def _loadNLP(self):
        import spacy

        try:
            self.nlp = spacy.load(self.spacyModel)
        except OSError:
            print(f"spaCy model '{self.spacyModel}' not found; falling back to "
                  f"en_core_web_sm (synonym matching disabled). "
                  f"Install it with: python -m spacy download {self.spacyModel}")
            self.nlp = spacy.load("en_core_web_sm")

        # Pre-lemmatise the vocabulary so query matching is a dict lookup per term,
        # and pre-compute normalised tag vectors for synonym matching (if the model
        # ships word vectors -- en_core_web_sm does not).
        self.lemmaToTags = {}
        vectors = []
        hasVectors = self.nlp.vocab.vectors_length > 0
        for i, tag in enumerate(self.vocab):
            doc = self.nlp(tag.lower())
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                self.lemmaToTags.setdefault(token.lemma_, []).append(i)
            if hasVectors:
                norm = doc.vector_norm
                vectors.append(doc.vector / norm if norm else np.zeros_like(doc.vector))

        self.tagVectors = np.stack(vectors).astype(np.float32) if hasVectors else None


class ClusteringSearch(FontSearch):
    def __init__(self, checkpoint=DEFAULT_FINETUNE, backbone=DEFAULT_BACKBONE,
                 dataset=None, embeddingName="all", device=device, visible=10, digits=8, k=8):
        self.device = device
        self.backbone = backbone
        self.embeddingName = embeddingName

        self.loadModels()
        self.dataset = dataset if dataset is not None else self.buildDataset()
        self.fontPathMap = self._buildPathMap()
        self.embeddings = self.embedFonts()

        self.visible = visible
        self.digits = digits
        self.k = k

        self.keys = list(self.embeddings.keys())
        self.matrix = torch.tensor(np.stack([self.embeddings[k] for k in self.keys]), dtype=torch.float32)

        bitsPer = int(round(math.log(self.k, 2)))
        self.pca = PCA(n_components=self.digits * bitsPer)
        transformed = self.pca.fit_transform(self.matrix)
        self.transformed = transformed.copy()

        transformed = np.reshape(transformed, [transformed.shape[0], -1, bitsPer])
        codes = np.zeros(transformed.shape[:-1], dtype=np.int32)
        for i in range(bitsPer):
            codes += (2 ** i) * (transformed[:, :, i] <= 0).astype(np.int32)
        self.codes = codes

        self.initializeLearner()
        
        self.rankings = {}
        self.results = []

    def initializeLearner(self):
        self.identified = np.full((self.digits), fill_value=np.nan)
        self.digit = 0
        self.getRepresentatives()

    def getRepresentatives(self):
        self.options = []

        if self.digit >= self.digits:
            return

        for i in range(self.k):
            code = self.identified.copy()
            code[self.digit] = i

            mask = ~np.isnan(code)
            target = code[mask].astype(np.int32)
            matches = np.all(self.codes[:, mask] == target, axis=1)
            chosenNames = np.array(self.keys)[matches][:self.visible]

            clusterSet = [(name, self.embeddings[name]) for name in chosenNames]

            self.options.append(clusterSet)

    def updateLocation(self, positive):
        self.identified[self.digit] = positive

        self.digit += 1

        self.getRepresentatives()

    def undo(self):
        """Reverse the most recent digit assignment so it can be reselected."""
        if self.digit == 0:
            return
        self.digit -= 1
        self.identified[self.digit] = np.nan
        self.getRepresentatives()

    def reset(self):
        """Clear every digit assignment and start back over at the first digit."""
        self.identified[:] = np.nan
        self.digit = 0
        self.getRepresentatives()

    @property
    def finished(self):
        return self.digit >= self.digits

    def matchedNames(self):
        """Every font name whose code matches the digits identified so far."""
        mask = ~np.isnan(self.identified)
        if not mask.any():
            return list(self.keys)
        target = self.identified[mask].astype(np.int32)
        matches = np.all(self.codes[:, mask] == target, axis=1)
        return list(np.array(self.keys)[matches])

    # ------------------------------------------------------------------ models
    def loadModels(self):
        """Load and stash the image / text models. Sets ``self.datasetConfig``."""
        self.imageModel, conf = ViT.load(self.backbone)

        self.config = conf
        self.datasetConfig = conf.dataset

        self.imageModel.eval().to(self.device)


class MeanderSearch(FontSearch):
    def __init__(self, checkpoint=DEFAULT_FINETUNE, backbone=DEFAULT_BACKBONE,
                 dataset=None, embeddingName="all", device=device, learningRate=3e-3):
        self.device = device
        self.backbone = backbone
        self.embeddingName = embeddingName

        self.loadModels()
        self.dataset = dataset if dataset is not None else self.buildDataset()
        self.fontPathMap = self._buildPathMap()
        self.embeddings = self.embedFonts()

        self.initializeLearner(learningRate)
        
        self.rankings = {}
        self.results = []

    def initializeLearner(self, learningRate):
        def cosineDistance(x, y):
            return 1 - (x @ y.t())
        
        self.location = torch.randn(self.embeddings[list(self.embeddings.keys())[0]].shape[0])
        self.location = nn.functional.normalize(self.location, dim=-1)
        self.location = nn.Parameter(self.location)

        self.getRepresentatives()

        self.learningRate = learningRate
        self.optimizer = torch.optim.SGD([self.location], lr=self.learningRate)
        self.objective = nn.TripletMarginWithDistanceLoss(distance_function=cosineDistance, margin=0.1)

    # Select embeddings perpendicular to the current location that are dissimilar to each other
    def getRepresentatives(self):
        self.options = []

        keys = np.array(list(self.embeddings.keys()))
        matrix = torch.tensor(np.stack([self.embeddings[k] for k in keys]), dtype=torch.float32)
        simMatrix = matrix @ matrix.t()

        key = nn.functional.normalize(self.location, dim=-1).t()
        scores = matrix @ key

        indices = torch.argsort(torch.abs(scores))

        names = keys[indices.numpy()]
        simMatrix = simMatrix[indices, :]
        simMatrix = simMatrix[:, indices]
        scores = scores[indices]

        check = 0

        for i in range(4):
            found = False
            while not found:
                name = names[check]

                # Check all current options to make sure we are also perpendicular
                # to the other options
                blocked = False
                for option in self.options:
                    location = list(names).index(option[0])
                    value = matrix[check, location]

                    if value > 0:
                        blocked = True
                        break

                if not blocked:
                    # Found a new option
                    self.options.append((name, self.embeddings[name]))
                    found = True

                # Remove embedding from list
                check += 1

    def updateLocation(self, positive, negative):
        normalized = nn.functional.normalize(self.location, dim=-1)
        pos = torch.tensor(positive[1], dtype=torch.float32)
        neg = torch.tensor(negative[1], dtype=torch.float32)
        loss = self.objective(normalized, pos, neg)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.getRepresentatives()

    # ------------------------------------------------------------------ models
    def loadModels(self):
        """Load and stash the image / text models. Sets ``self.datasetConfig``."""
        self.imageModel, conf = ViT.load(self.backbone)

        self.config = conf
        self.datasetConfig = conf.dataset

        self.imageModel.eval().to(self.device)

    @torch.no_grad()
    def encodeQuery(self, query):
        keys = list(self.embeddings.keys())
        matrix = torch.tensor(np.stack([self.embeddings[k] for k in keys]), dtype=torch.float32)
        scores = matrix @ nn.functional.normalize(self.location, dim=-1).t()
        return {keys[i]: scores[i].item() for i in range(len(keys))}


class WalkSearch(FontSearch):
    def __init__(self, checkpoint=DEFAULT_FINETUNE, backbone=DEFAULT_BACKBONE,
                 dataset=None, embeddingName="all", device=device,
                 priorVar=1.0, obsVar=0.25, obsMargin=1.0, numOptions=4):
        self.device = device
        self.backbone = backbone
        self.embeddingName = embeddingName
        self.priorVar = priorVar
        self.obsVar = obsVar
        self.obsMargin = obsMargin
        self.numOptions = numOptions

        self.loadModels()
        self.dataset = dataset if dataset is not None else self.buildDataset()
        self.fontPathMap = self._buildPathMap()
        self.embeddings = self.embedFonts()

        self.matrixKeys = list(self.embeddings.keys())
        self.matrix = torch.tensor(np.stack([self.embeddings[k] for k in self.matrixKeys]), dtype=torch.float32)

        centered = self.matrix - self.matrix.mean(dim=0)
        self.corpusCov = (centered.t() @ centered) / self.matrix.shape[0]

        DEff = 75  # start here, tune against the participation ratio you measure above

        eigvals, eigvecs = torch.linalg.eigh(self.corpusCov)
        topEigvecs = eigvecs[:, -DEff:]
        self.corpusCov = topEigvecs @ topEigvecs.t()

        self.initializeLearner()

        self.rankings = {}
        self.results = []

    def initializeLearner(self, priorVar=None, obsVar=None, obsMargin=None, numOptions=None):
        if priorVar is not None:
            self.priorVar = priorVar
        if obsVar is not None:
            self.obsVar = obsVar
        if obsMargin is not None:
            self.obsMargin = obsMargin
        if numOptions is not None:
            self.numOptions = numOptions

        dim = self.matrix.shape[1]
        self.location = nn.functional.normalize(torch.randn(dim), dim=-1)
        self.covariance = self.priorVar * self.corpusCov

        self.getRepresentatives()

    # Select candidates at the extremes of whatever directions the current
    # belief is still most uncertain about.
    # def getRepresentatives(self):
    #     eigvals, eigvecs = torch.linalg.eigh(self.covariance)
    #     topDirs = eigvecs[:, -(self.numOptions // 2):]

    #     proj = self.matrix @ topDirs

    #     chosen = []
    #     for d in range(topDirs.shape[1]):
    #         order = torch.argsort(proj[:, d])
    #         chosen.append(order[0].item())
    #         chosen.append(order[-1].item())
    #     chosen = list(dict.fromkeys(chosen))[:self.numOptions]

    #     self.options = [(self.matrixKeys[i], self.matrix[i].numpy()) for i in chosen]

    def getRepresentatives(self):
        location = nn.functional.normalize(self.location, dim=-1)
        P = torch.eye(location.shape[0]) - torch.outer(location, location)
        covPerp = P @ self.covariance @ P

        eigvals, eigvecs = torch.linalg.eigh(covPerp)
        topDirs = eigvecs[:, -(self.numOptions // 2):]

        proj = self.matrix @ topDirs

        chosen = []
        for d in range(topDirs.shape[1]):
            order = torch.argsort(proj[:, d])
            chosen.append(order[0].item())
            chosen.append(order[-1].item())
        chosen = list(dict.fromkeys(chosen))[:self.numOptions]

        self.options = [(self.matrixKeys[i], self.matrix[i].numpy()) for i in chosen]

    def updateLocation(self, positive, negative):
        optionsMatrix = nn.functional.normalize(
            torch.tensor(np.stack([o[1] for o in self.options]), dtype=torch.float32), dim=-1)

        p = nn.functional.normalize(torch.tensor(positive[1], dtype=torch.float32), dim=-1)
        n = nn.functional.normalize(torch.tensor(negative[1], dtype=torch.float32), dim=-1)
        axis = p - n

        weights = optionsMatrix @ axis      # each shown option's correlation to the p/n axis
        a = weights @ optionsMatrix         # M(p - n): correlation-weighted centroid direction

        y = self.obsMargin - a @ self.location
        s = a @ self.covariance @ a + self.obsVar
        gain = (self.covariance @ a) / s

        self.location = nn.functional.normalize(self.location + gain * y, dim=-1)
        self.covariance = self.covariance - torch.outer(gain, a) @ self.covariance

        self.getRepresentatives()

    # ------------------------------------------------------------------ models
    def loadModels(self):
        self.imageModel, conf = ViT.load(self.backbone)
        self.config = conf
        self.datasetConfig = conf.dataset
        self.imageModel.eval().to(self.device)

    @torch.no_grad()
    def encodeQuery(self, query):
        scores = self.matrix @ nn.functional.normalize(self.location, dim=-1)
        return {self.matrixKeys[i]: scores[i].item() for i in range(len(self.matrixKeys))}


class GridFeedbackSearch(FontSearch):
    """
    Interactive Grid-Feedback Retrieval.

    No text query, no image upload. The user is shown a grid of ``m`` fonts each
    round and picks whichever (if any) are close to what they want -- including
    picking none, which is itself a negative label for every shown font. Every
    round's labels accumulate into ``L`` and are refit from scratch as an
    L2-regularised logistic regression over a whitened PCA subspace of raw
    (pre-projection) backbone CLS features; the next grid is just the top-``m``
    unshown fonts by the fitted score. See the design doc, section 3, for why
    this beats text/tag search (insufficient channel capacity, not a bad
    embedding space) and why whitening is load-bearing (~60 labels in ~56 dims
    is underdetermined otherwise).

    Offline (once per corpus, cached under ``embeddings/``):
        raw per-font features -> centre -> PCA -> whiten -> ``Z``
        k-means(Z, m) -> medoid of each cluster -> seed grid

    Online (per session, held on ``self``):
        round r: show ``self.options`` -> ``selectFonts(picks)`` labels them,
        refits ``self.weight``, and replaces ``self.options`` with the next grid.
    """

    def __init__(self, backbone=DEFAULT_BACKBONE, dataset=None, embeddingName="gridRaw",
                 device=device, m=20, D=None, regularization=1.0):
        self.device = device
        self.backbone = backbone
        self.embeddingName = embeddingName
        self.m = m
        self.regularization = regularization

        self.loadModels()
        self.dataset = dataset if dataset is not None else self.buildDataset()
        self.fontPathMap = self._buildPathMap()
        self.rawEmbeddings = self.embedFonts()

        self.keys = np.array(list(self.rawEmbeddings.keys()))
        self.nameToIndex = {name: i for i, name in enumerate(self.keys)}
        rawMatrix = np.stack([self.rawEmbeddings[k] for k in self.keys], axis=0)

        self.D = D if D is not None else self._participationRatio(rawMatrix)
        self.D = max(1, min(self.D, rawMatrix.shape[0] - 1, rawMatrix.shape[1]))

        self._fitProjection(rawMatrix)
        self.seed = self._seedGrid()

        self.reset()

    # ------------------------------------------------------------------ models
    def loadModels(self):
        """Only the pretraining backbone is needed -- there is no finetuned checkpoint."""
        self.imageModel, conf = ViT.load(self.backbone)
        self.config = conf
        self.datasetConfig = conf.dataset
        self.imageModel.eval().to(self.device)

    # ------------------------------------------------------------- raw features
    @torch.no_grad()
    def _rawFeatures(self, batch):
        """
        The backbone's CLS token *before* the pretraining classifier head --
        ``h`` rather than ``z``. ``z`` is trained to be invariant to the SSL
        augmentations (scale/blur/contrast), which is exactly the style signal
        this method needs, so the classifier/projection is deliberately skipped.
        """
        x = batch.permute(0, 3, 1, 2)
        x = self.imageModel.patching(x)
        x = torch.cat([self.imageModel.clsToken.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.imageModel.transformer(x)
        return x[:, 0]

    @torch.no_grad()
    def embedFonts(self):
        """
        Per-font mean raw CLS feature across its latin glyphs -- unnormalised,
        unlike ``generateEmbeddings`` (see design doc step 1). Cached under
        ``embeddings/<name>.json`` like the other search backends.
        """
        os.makedirs("embeddings", exist_ok=True)
        path = os.path.join("embeddings", f"{self.embeddingName}.json")
        if os.path.exists(path):
            with open(path, "r") as file:
                return {key: np.array(value) for key, value in json.load(file).items()}

        lookup = {(self.dataset.names[i], self.dataset.letters[i]): self.dataset.paths[i]
                  for i in range(len(self.dataset.names))}
        names = np.unique(self.dataset.names)

        embeddings = {}
        for n, name in enumerate(names):
            images = []
            broken = False
            for letter in latinCharacters:
                if (name, letter) not in lookup:
                    broken = True
                    break
                _, image = loadImage(lookup[(name, letter)])
                if image is None:
                    broken = True
                    break
                images.append(torch.tensor(image, dtype=torch.float32))

            if broken:
                continue

            batch = torch.stack(images, dim=0).unsqueeze(-1).to(self.device)
            features = self._rawFeatures(batch).mean(dim=0)
            embeddings[name] = features.cpu().numpy()

            print(f"\r{n + 1}/{len(names)} raw grid features extracted", end="")

        print()
        with open(path, "w+") as file:
            json.dump({key: value.tolist() for key, value in embeddings.items()}, file)
        return embeddings

    # --------------------------------------------------------- offline PCA/whiten
    @staticmethod
    def _participationRatio(V):
        """(sum(eigvals))^2 / sum(eigvals^2) -- effective dimensionality of the spectrum."""
        centered = V - V.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
        ratio = (eigvals.sum() ** 2) / (np.square(eigvals).sum() + 1e-12)
        return int(round(ratio))

    def _fitProjection(self, V):
        """Centre -> PCA(D) -> whiten, cached alongside the raw feature cache."""
        cachePath = os.path.join("embeddings", f"{self.embeddingName}_projection.npz")
        if os.path.exists(cachePath):
            cached = np.load(cachePath)
            if int(cached["D"]) == self.D:
                self.mean = cached["mean"]
                self.components = cached["components"]
                self.std = cached["std"]
                self.Z = cached["Z"]
                return

        self.mean = V.mean(axis=0)
        centered = V - self.mean

        pca = PCA(n_components=self.D)
        projected = pca.fit_transform(centered)
        self.components = pca.components_

        self.std = projected.std(axis=0)
        self.std[self.std < 1e-8] = 1e-8
        self.Z = projected / self.std

        os.makedirs("embeddings", exist_ok=True)
        np.savez(cachePath, mean=self.mean, components=self.components,
                 std=self.std, Z=self.Z, D=self.D)

    def _seedGrid(self):
        """k-means(Z, m) -> index of the medoid (nearest-to-centroid member) of each cluster."""
        clusters = min(self.m, self.Z.shape[0])
        kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=0).fit(self.Z)

        seed = []
        for c in range(clusters):
            members = np.where(kmeans.labels_ == c)[0]
            if len(members) == 0:
                continue
            distances = np.linalg.norm(self.Z[members] - kmeans.cluster_centers_[c], axis=1)
            seed.append(members[np.argmin(distances)])
        return np.array(seed)

    # ------------------------------------------------------------- online belief
    def reset(self):
        """Clear all labels and belief, and reissue the seed grid as round 1."""
        self.labeledIndices = []
        self.labels = []
        self.shown = set()
        self.weight = np.zeros(self.D)
        self.round = 1
        self.finished = False
        self.terminalName = None

        self.options = self._namesFor(self.seed)
        self.shown.update(int(i) for i in self.seed)

    def _namesFor(self, indices):
        return [(self.keys[i], self.Z[i]) for i in indices]

    def accept(self, name):
        """User accepted a font straight out of the grid -- terminate the session."""
        self.finished = True
        self.terminalName = name

    def selectFonts(self, selectedNames):
        """
        Submit the user's picks from the current grid (``self.options``) and
        advance to the next round. ``selectedNames`` may be empty -- "nothing
        here is close" is itself a negative label for every shown font, per the
        design doc's free-selection model.
        """
        selected = set(selectedNames)
        for name, _ in self.options:
            index = self.nameToIndex[name]
            self.labeledIndices.append(index)
            self.labels.append(1 if name in selected else 0)

        self._fitWeight()
        self.options = self._nextGrid()
        self.round += 1
        return self.options

    def _fitWeight(self):
        labels = np.array(self.labels)
        if len(np.unique(labels)) < 2:
            # No contrast yet (all-positive or all-negative so far): fall back
            # to the centroid of whatever has been marked positive.
            positives = [self.Z[i] for i, y in zip(self.labeledIndices, self.labels) if y == 1]
            self.weight = np.mean(positives, axis=0) if positives else np.zeros(self.D)
            return

        Z = self.Z[self.labeledIndices]
        signed = np.where(labels == 1, 1.0, -1.0)
        self.weight = self._fitLogistic(Z, signed, self.regularization)

    @staticmethod
    def _fitLogistic(Z, y, regularization, iterations=200):
        """
        L2-regularised logistic regression, refit from scratch every round:
            argmin_w  sum_j log(1 + exp(-y_j w.z_j)) + (lambda/2)||w||^2
        Cheap at D ~ 50-60 and at most a few hundred labels.
        """
        def lossAndGrad(w):
            margins = y * (Z @ w)
            loss = np.logaddexp(0, -margins).sum() + 0.5 * regularization * (w @ w)
            p = 1.0 / (1.0 + np.exp(margins))
            grad = -(y * p) @ Z + regularization * w
            return loss, grad

        result = minimize(lossAndGrad, np.zeros(Z.shape[1]), jac=True,
                           method="L-BFGS-B", options={"maxiter": iterations})
        return result.x

    def _nextGrid(self):
        """Top-m unshown fonts by the current belief: argsort(-mu over corpus \\ shown)[:m]."""
        scores = self.Z @ self.weight
        order = np.argsort(-scores)

        chosen = []
        for index in order:
            if int(index) in self.shown:
                continue
            chosen.append(index)
            if len(chosen) >= self.m:
                break

        self.shown.update(int(i) for i in chosen)
        return self._namesFor(chosen)