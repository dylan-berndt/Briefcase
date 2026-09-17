# Text -> Visual Font Embedding Diffusion Experiment

Tests whether a diffusion model can learn `P(visual font embedding | text embedding)`.

## Current pipeline (targets `embeddings/all.json`, the staging-branch corpus)

```
experiments/embed_text_local.py  # -> embeddings/sentenceQueries_<model>.pkl  {fontKey: [nQueries, dim]}
                                  #    run LOCALLY (not this sandbox) -- see Compute note
experiments/train.py             # -> checkpoints/diffusion/{checkpoint.pt, stats.npz, config.json, test_pairs.json}
experiments/evaluate.py          # recall@k over held-out queries
```

`embeddings/all.json` (39,421 fonts, LFS-tracked) and `results/fontQueriesV2.json`
(18,940 fonts x up to 16 Gemma-generated queries each) already exist on `staging`
and are consumed directly by `experiments/dataset.py` -- no font rasterization or
ViT inference needed for this path.

## Design decisions / assumptions

- **Visual embedding**: `embeddings/all.json` -- the *contrastively-finetuned*
  `ViTEmbedder` head's pooled output, not the raw pretrained ViT CLS token.
  (An earlier version of this experiment targeted the raw CLS output instead;
  see "Effective dimension" below for why that changed.) `embed_fonts.py` /
  `effective_dimension.py` / `character_variance.py` / `pooling_bug_check.py`
  still exist and operate on the raw-CLS space (`embeddings/font_vit_embeddings.json`)
  for that comparison, but are no longer part of the main training path.
- **Text encoder**: swappable via `--model`/`--sentenceModel` (must match between
  `embed_text_local.py` and `train.py`/`evaluate.py`), defaulting to
  `BAAI/bge-large-en-v1.5`. BGE models pool via the `[CLS]` token by default
  (`1_Pooling/config.json` has `pooling_mode_cls_token: true`), rather than the
  mean-pooling most `sentence-transformers/all-*` models use.
- **Query matching**: `embed_text_local.py`/`dataset.py` reuse
  `experiments/text-searches/trainTextMLP.py`'s exact matching convention:
  `embeddings/all.json` is keyed by per-style-variant render name (e.g.
  `" Really Petshop Italic  Really Petshop Italic"`), `fontQueriesV2.json` by
  base family name (`"Really Petshop"`); match via longest-prefix trie, and
  when a family has multiple matching style variants keep the shortest-named
  one as the canonical "regular" embedding.
- **Train/test split holds out queries, not fonts** (`dataset.splitQueryCache`,
  same convention as `trainTextMLP.py`): each font's own queries are split
  independently (default 80/20), so every font appears in training, and every
  font with more than one query also has held-out queries in test.
- **Diffusion model**: a plain residual MLP (`experiments/diffusion.py:
  DiffusionMLP`) conditioned on a sinusoidal timestep embedding and the raw
  text embedding (no visual information at all besides the noised target) --
  a standard DDPM (linear beta schedule, 1000 steps by default) predicting
  the added noise. Visual embeddings are standardized (per-dimension
  zero-mean/unit-variance, fit on the training fonts only) before diffusion
  so `x_T ~ N(0, I)` is a sane prior; `stats.npz` stores the fit for eval-time
  denormalization.
- **Recall@k evaluation** (`evaluate.py`): for each held-out test query, runs
  `--samplesPerQuery` (default 8) *independent full reverse-diffusion
  processes* from fresh Gaussian noise (no shortcuts), takes the best cosine
  similarity across those samples against every known font's visual
  embedding (fonts are never held out, so the true font is always a
  candidate), and reports whether the true font lands in the top k for
  k in {1, 5, 10, 50, 100}.
- **Verified working** (`experiments/*` as of this commit): smoke-tested
  end-to-end on a 300-font / bge-small subset -- `embed_text_local.py` ->
  `train.py` (3 epochs) -> `evaluate.py` all ran cleanly. Recall@k was 0 at
  that scale/epoch count, as expected (mechanical check only, not a real result).

## Effective dimension of the visual embedding space

`embeddings/font_vit_embeddings.json` (raw pretrained ViT CLS, mean-pooled over
a-z, ~39.5k fonts) has an effective rank of **~9.5 out of 512** nominal dims
(`effective_dimension.py`) -- true even for individual, unpooled, per-letter
embeddings before any pooling happens (`character_variance.py` rules out
pooling as the cause). `embeddings/all.json` (the contrastively-finetuned
`ViTEmbedder` space) has an effective rank of **~57.6 out of 512** -- about
6x higher, consistent with contrastive/InfoNCE training being what actually
spreads font-identity information across the embedding space; the raw
pretrain objective (reconstruction + character classification) never
incentivizes that. Still fairly anisotropic in absolute terms (~11% of
nominal dims used) -- CLAUDE.md's own "All-but-the-Top" suggestion is a
reasonable inference-time fix worth trying independent of this experiment.

Both embedding files also have the same normalization bug (`pooling_bug_check.py`):
`utils/embeddings.generateEmbeddings` normalizes each per-letter embedding
before mean-pooling but never renormalizes the pooled result, so stored font
vectors don't sit exactly on the unit hypersphere (norms observed: 0.70-1.00
for font_vit_embeddings.json, 0.72-1.00 for all.json) and cosine similarity
against them isn't exactly the average pairwise letter-cosine it's meant to
approximate. Not yet fixed upstream as of this commit.

## Compute note

Font/ViT-embedding generation and the CPU-only DDPM training loop are fine on
a modest sandbox, but text-encoding tens/hundreds of thousands of queries with
a large sentence-transformer model (e.g. bge-large) is not -- run
`embed_text_local.py` locally (ideally with a GPU) rather than in a CPU-only
sandbox; a full bge-large pass over ~380k queries measured at ~3s/batch
(batch 64) there, i.e. hours.
