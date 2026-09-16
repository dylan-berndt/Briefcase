# Text -> Visual Font Embedding Diffusion Experiment

Tests whether a diffusion model can learn `P(visual font embedding | text embedding)`,
using the repo's pretrained font ViT (`checkpoints/pretrain/best`) as the source of
"ground truth" visual embeddings and the pregenerated font-search queries
(`results/fontQueries.json`, `results/shortQueries.json`) as text.

## Pipeline

```
experiments/setup_data.sh   # venv + Google Fonts + DaFont + MyFonts ("dataset")
experiments/embed_fonts.py  # -> embeddings/font_vit_embeddings.json   {fontKey: [512]}
experiments/embed_text.py   # -> embeddings/text_query_embeddings.json {query: [1024]}
                             # -> embeddings/query_font_pairs.json     [{font, query, source}]
experiments/train.py        # -> checkpoints/diffusion/{checkpoint.pt, stats.npz, config.json, test_pairs.json}
experiments/evaluate.py     # recall@k over held-out queries
```

Run `run_pipeline.sh` to do the first three steps in one shot.

## Design decisions / assumptions

- **Visual embedding**: the raw pretrained ViT's CLS-token output
  (`checkpoints/pretrain/best`), *not* the contrastively-finetuned
  `ViTEmbedder` head. This keeps the diffusion task a clean probe of the
  modality gap rather than something already partially aligned to text
  during finetuning. Per-font embedding = L2-normalized per-letter CLS
  embeddings, mean-pooled over lowercase a-z (same convention
  `utils/embeddings.generateEmbeddings` already used elsewhere in the repo).
- **Font sources**: Google Fonts + DaFont + MyFonts ("dataset", the
  Rochester/gdown corpus), matching `configs/vit.json`'s original pretraining
  sources. `embed_fonts.py` only rasterizes lowercase a-z per font (26 glyphs)
  rather than the ~200-character set `collectFontSetPaths` normally produces,
  since that's all `generateEmbeddings` averages over -- this keeps disk usage
  down by roughly 8x with no effect on the resulting embeddings.
- **Text encoder**: `BAAI/bge-large-en-v1.5` via `sentence-transformers`.
  BGE models pool via the `[CLS]` token by default (`1_Pooling/config.json`
  has `pooling_mode_cls_token: true`), rather than the mean-pooling most
  `sentence-transformers/all-*` models use.
- **Query matching**: `fontQueries.json`/`shortQueries.json` are keyed by
  bare family/slug names (e.g. `"Nokora"`), while visual embeddings are keyed
  by `"{family} {style}"` (e.g. `"Open Sans Regular"`, from `font.getname()`).
  `embed_text.py` matches them with the same longest-prefix trie lookup
  `CombinedQueryData` uses in `utils/querying.py`. Fonts that only exist in
  the query files (e.g. described from MyFonts descriptions but never
  successfully rasterized) are simply dropped from the pairing.
- **Query set union**: the two query files are concatenated per font
  (`fontQueries` + `shortQueries`, tagged by `source` in
  `query_font_pairs.json` in case you want to analyze them separately later).
- **Train/test split holds out queries, not fonts** (`dataset.splitPairs`):
  each font's own queries are split independently (default 80/20), so every
  font with a query appears in training, and every font with more than one
  query also has held-out queries in test. This matches the requirement to
  evaluate on unseen prompts while keeping the full font vocabulary visible
  during training.
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

## Compute note

This was kicked off on a CPU-only, ~30GB-disk sandbox. Downloading/rasterizing
all three font corpora and running full-reverse-diffusion recall@k eval
(`samplesPerQuery` x `timesteps` model calls per query) is slow without a GPU.
If the sandbox run doesn't finish, everything above is meant to be run
locally as-is: `bash experiments/setup_data.sh && python3 experiments/embed_fonts.py
&& python3 experiments/embed_text.py && python3 experiments/train.py && python3
experiments/evaluate.py`.
