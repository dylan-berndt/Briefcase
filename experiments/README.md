# Text -> Visual Font Embedding Diffusion Experiment

Tests whether a diffusion model can learn `P(visual font embedding | text embedding)`.

## Current pipeline (targets `embeddings/all.json`, the staging-branch corpus)

```
experiments/embed_text_local.py  # -> embeddings/sentenceQueries_<model>.pkl  {fontKey: [nQueries, dim]}
                                  #    run LOCALLY (not this sandbox) -- see Compute note
experiments/train.py             # -> checkpoints/diffusion/{checkpoint.pt, whitener.npz, config.json, test_pairs.json}
experiments/evaluate.py          # recall@k over held-out queries
experiments/diagnose_conditioning.py     # generated-vs-true pairwise similarity structure
experiments/diagnose_text_ablation.py    # denoising loss with real vs. shuffled text, by noise level
experiments/diagnose_manifold_realism.py # does generation look like ANY real font, right or wrong
```

`embeddings/all.json` (39,421 fonts, LFS-tracked) and `results/fontQueriesV2.json`
(18,940 fonts x up to 16 Gemma-generated queries each) already exist on `staging`
and are consumed directly by `experiments/dataset.py` -- no font rasterization or
ViT inference needed for this path.

## Design decisions / assumptions

- **Visual embedding**: `embeddings/all.json`, not the raw pretrained ViT CLS
  token an earlier version of this experiment targeted (`embeddings/font_vit_embeddings.json`,
  still around for the effective-dimension comparison below). Note: `all.json`
  is *not* the `ViTEmbedder` contrastive-finetuning output -- that's a separate
  file, `embeddings/allText.json` -- `all.json` is on the raw/pretrain side too
  (per the user, from the newer 12-layer pretrain checkpoint), just a better
  model than the one `font_vit_embeddings.json` used. `embed_fonts.py` /
  `effective_dimension.py` / `character_variance.py` / `pooling_bug_check.py`
  still exist and operate on `font_vit_embeddings.json` for that comparison,
  but are no longer part of the main training path.
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
  the added noise.
- **The diffusion process operates in a 64-d PCA-whitened subspace, not the
  raw 512-d ambient space** (`dataset.PCAWhitener`, fit on training fonts
  only; `--pcaDim`, default 64). This replaced simple per-dimension
  standardization after diagnosing why early full-scale runs generated
  samples that didn't resemble any real font at all (see "Diagnostic
  findings" below): `embeddings/all.json` has an effective rank of only
  ~57.6/512, so the other ~450 ambient dimensions carry almost no real
  signal, and small per-dimension prediction errors summed across all of
  them can produce a large, essentially random perturbation that swamps the
  real signal. Same idea `utils/search.py`'s `GridFeedbackSearch` already
  uses (centre -> PCA -> whiten) for the same anisotropic embedding space,
  applied here to the diffusion target instead of a search index. 64 was
  chosen to sit a bit above the measured 57.6 effective rank, for headroom.
  `whitener.npz` stores the fit; `evaluate.py`/`diagnose_*.py` project
  generated samples back to the 512-d ambient space (`inverseTransform`)
  before computing cosine similarity against the corpus, so recall@k and
  the diagnostics stay comparable to the earlier full-ambient-space runs.
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

## Diagnostic findings (first full-scale run, pre-PCA-whitening)

A full run (17,056 matched fonts, 221,888 train / 51,164 test queries, bge-large,
60 epochs, loss 0.433 -> 0.063) got recall@k barely above random chance
(recall@5 0.0004 vs. a 0.000127 random baseline at 39,421 candidates -- roughly
2.4-3.3x chance across k, but tiny in absolute terms). Three diagnostics,
in the order run:

1. **`diagnose_conditioning.py`** (generated-vs-true pairwise similarity
   structure across distinct-font queries): correlation ~0, and generated
   samples averaged only ~0.13 cosine to the corpus-wide mean font vs. real
   fonts' ~0.75 -- with very high pairwise-similarity variance (std ~0.68,
   vs. ~0.044 expected for genuinely random 512-d vectors), the signature of
   a low-dimensional subspace uncorrelated with the real manifold. Identical
   at 50 and 1000 sampling steps, ruling out ancestral-respacing noise as
   the cause.
2. **`diagnose_text_ablation.py`** (denoising loss with real vs. shuffled
   text, holding x0/noise/t fixed, by timestep bucket): real text beat
   shuffled text at every noise level (~1-7% relative, peaking mid-schedule)
   -- ruling out full conditioning collapse. The effect is real, just modest.
3. **`diagnose_manifold_realism.py`** (nearest-neighbor cosine of generated
   samples against the full corpus, vs. real fonts' own leave-one-out
   nearest-neighbor baseline): real fonts sit in an extremely tight cluster
   (mean nearest-other-real cosine ~0.96); generated samples' best match
   anywhere in the corpus averaged only ~0.40. Most generated output doesn't
   resemble *any* real font, correct or not.

Read together: conditioning has a real but modest effect, and generation
quality (not landing near the real, very concentrated manifold at all) looks
like the dominant problem -- consistent with small per-dimension errors,
replicated across the ~450 near-degenerate ambient dimensions
`embeddings/all.json` actually has (see effective dimension below),
dominating the output direction. Not yet re-run against the PCA-whitened
version above to confirm the fix.

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
