# Embedding geometry: why does font text search fail where general image search succeeds?

## The question

This project's font search (`experiments/diffusion-searches/`) has repeatedly measured that a short
text query can't reliably pick out one specific font from the ~39,421-font corpus (single-shot
R@1≈0%), even against real human-written descriptions. Yet ordinary image search ("a red bicycle
leaning against a brick wall") works well on corpora that are far larger and, on the surface, far
more complex than a font corpus. The font visual embeddings themselves look well-behaved (a real
pretrained ViT, sensible effective dimensionality once you know to whiten/PCA it). So what's
actually different?

This experiment reproduces the SAME kind of decoupled setup the font project uses -- an image/visual
encoder that was never jointly trained with a text encoder, paired with a generic, off-the-shelf text
encoder that was never finetuned on this specific retrieval task -- but on a domain (natural images
with captions) where retrieval is known to work reasonably well, so the two can be compared side by
side on the same geometric diagnostics and the same retrieval method.

## Why the encoders have to be decoupled, deliberately

If we used CLIP's own image and text towers, we'd be comparing "two encoders that were jointly,
contrastively trained to align with each other" against the font project's "two encoders trained
completely independently, never seeing the other modality." That's not a fair comparison -- CLIP's
image-text alignment is *engineered in* by construction; ours isn't. So this experiment specifically
uses:

- **Image side: DINOv2** (`facebookresearch/dinov2`, ViT-S/14) -- purely self-supervised, never saw a
  single word of text during training. This is the closest natural-image analogue to this project's
  own font ViT (pretrained on glyph reconstruction, also never saw text).
- **Text side: BGE-large-en-v1.5** -- the exact same sentence-transformer model already used
  throughout `experiments/diffusion-searches/` for font queries. It was never finetuned on
  image-caption retrieval, exactly mirroring how it was never finetuned on font retrieval either.

Neither encoder has ever seen the other modality or this specific task. Any alignment between them
has to come from the paired data itself, not from joint training -- which is exactly the situation
the font project is in.

## The known-working reference method: ASIF

Rather than inventing a new alignment technique from scratch, this uses **ASIF** (Norelli et al.,
NeurIPS 2023, ["ASIF: Coupled Data Turns Unimodal Models to Multimodal Without
Training"](https://arxiv.org/abs/2210.01738)) -- a *training-free* method built for exactly this
setup: independently pretrained, never-jointly-trained unimodal encoders, aligned using only a modest
number of paired examples. The paper's own experiments used DINO (self-supervised vision) + a
pretrained SentenceTransformer (never finetuned on retrieval) and got zero-shot classification
competitive with CLIP using ~250x less paired data. Reference implementation:
[github.com/noranta4/ASIF](https://github.com/noranta4/ASIF).

**The method** (`asif_retrieval.py`): pick N "anchor" pairs from the paired dataset. For any new
image, its *relative representation* is the vector of its cosine similarities to the N anchor
IMAGES (in image-embedding space). For any new text, its relative representation is the vector of
its cosine similarities to the N anchor CAPTIONS (in text-embedding space, a totally different
space). Because the anchors are paired, both relative representations end up indexed by the *same*
N anchor-pair ids -- so even though they came from unrelated encoders, they're now directly
comparable vectors. Sparsify (zero everything but the top-k similarities) and exponentiate (raise to
a power p) to sharpen the signal, then rank candidates by cosine similarity between relative
representations. No training, no gradient steps -- just paired data and two frozen encoders.

## What's being compared

Two corpora, run through the *same* code path (`geometry_comparison.py`, `asif_retrieval.py`):

| | Flickr8k (reference) | Fonts (this project) |
|---|---|---|
| items | 8,091 images | 39,421 fonts |
| paired text | 5 human captions/image | up to 16 Gemma-generated queries/font |
| image/visual encoder | DINOv2 ViT-S/14 (self-supervised) | project's own pretrained ViT (reconstruction-supervised) |
| text encoder | BGE-large-en-v1.5 | BGE-large-en-v1.5 (same model) |
| jointly trained? | no | no |

Diagnostics computed identically for both: effective dimensionality (participation ratio) of each
embedding space, nearest-neighbor cosine similarity distribution (corpus density -- this is the one
most directly implicated by the font investigation's own finding that real fonts sit at ~0.96 cosine
to their nearest neighbor), and ASIF-style retrieval recall@k.

## Files

- `download_flickr8k.py` -- fetches the dataset (community GitHub mirror; the original host is down).
- `embed_flickr8k.py` -- embeds images (DINOv2) and captions (BGE), caches to `data/`.
- `geometry_comparison.py` -- effective-dimension and nearest-neighbor-density diagnostics, both corpora.
- `asif_retrieval.py` -- ASIF-style relative-representation retrieval, both corpora, recall@k.

## Results

### Geometry (`geometry_comparison.py`)

| | Flickr8k image (DINOv2) | Flickr8k text (BGE) | Font visual (project ViT) | Font text (BGE) |
|---|---|---|---|---|
| n | 8,091 | 40,460 | 39,421 | 273,052 |
| dim | 384 | 1024 | 512 | 1024 |
| participation ratio | 106.1 | 59.6 | 57.6 | 53.8 |
| nearest-neighbor cosine (mean) | **0.626** | **0.850** | **0.962** | **0.938** |

### Retrieval (`asif_retrieval.py`, matched corpus size, n=8,091 items both sides, same anchors=6,000)

| | Flickr8k (reference) | Fonts |
|---|---|---|
| recall@1 | 4.88% | 0.20% |
| recall@5 | 22.56% | 1.14% |
| recall@10 | 35.12% | 2.10% |
| recall@50 | 66.72% | 8.34% |
| recall@100 | 79.40% | 14.51% |
| median rank | 22 / 8,091 | 764 / 8,091 |

### Conclusion

Effective dimensionality (participation ratio) is actually comparable between the two domains --
fonts aren't using an unusually small or degenerate fraction of their embedding space. The real,
large, consistent difference is **nearest-neighbor density, in both modalities**: font visual
embeddings sit at 0.962 mean cosine to their nearest neighbor vs. 0.626 for natural images (fonts
are ~10x tighter by a 1-cosine distance proxy), and even font TEXT DESCRIPTIONS are more mutually
similar (0.938) than natural image captions (0.850). Running the exact same training-free alignment
method (ASIF), on matched corpus size, with the identical text encoder, retrieval is ~20-25x worse
on fonts at every k.

Nothing is broken or unusually badly-behaved about the font embeddings -- the domain itself is
fundamentally less discriminable in both modalities than natural images. Natural photos combine many
independently-verifiable, language-describable attributes (object identity, color, count, pose,
background, spatial relation) that compose to narrow the space combinatorially. Font style is a much
lower-dimensional, more continuous design space, and the vocabulary people use to describe it
(a few dozen adjectives like "elegant," "modern," "bold," "script") is reused across thousands of
genuinely-different-but-similarly-describable fonts -- so short text queries carry far less
discriminative information here than they do for natural images, independent of embedding quality,
training method, or corpus size. This is the same conclusion the diffusion-searches investigation's
interactive branching work was already built around; this experiment gives it a controlled,
quantitative confirmation against a known-working reference rather than resting on it as an
assumption.
