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
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .config import Config
from .vit import ViT, ViTEmbedder
from .querying import CombinedQueryData, CLIPTextEmbedder
from .embeddings import generateEmbeddings, latinCharacters
from .pretraining import device, loadImage
from .loaders.description import Description

__all__ = ["FontSearch", "TagSearch", "MeanderSearch"]


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

        def cosineDistance(x, y):
            return 1 - (x @ y.t())
        
        self.location = nn.Parameter(torch.randn(self.embeddings[list(self.embeddings.keys())[0]].shape[0]))

        self.getRepresentatives()

        self.learningRate = learningRate
        self.optimizer = torch.optim.SGD([self.location], lr=self.learningRate)
        self.objective = nn.TripletMarginWithDistanceLoss(distance_function=cosineDistance, margin=0.1)
        
        self.rankings = {}
        self.results = []

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
