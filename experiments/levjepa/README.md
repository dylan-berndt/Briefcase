# LeVJEPA-style pretraining on font glyphs

Adapts LeVJEPA (Kuhn et al., arXiv 2608.27395; SIGReg from LeJEPA, arXiv 2511.08544) to fonts:

- **Frames are glyphs.** Each font is 52 glyphs (a–z, A–Z), each 6×6 patches of 8px. Tokens carry 3-axis rotary
  positions (letter, row, col).
- **Global view:** every glyph of the font, with 10% of tokens kept (75% of them from ink patches).
- **Local views:** 4 views, each a 24px crop shared across glyphs, also 10% of tokens kept.
- **Loss:** `L_inv + 0.02 * SIGReg`. There is no predictor, target encoder or negatives.

All sources use the MyFonts rendering (tight crop, scale-to-fit).

| file | does |
|---|---|
| `build_glyph_cache.py` | builds `dataset/levjepa/` (40,112 fonts, uint8 `[N, 52, 48, 48]` memmap + presence mask), rendering Google/DaFont through the MyFonts pipeline |
| `train.py` | pretraining; `--supervise` restarts and resumes after crashes; logs `LeVJEPA/*` to wandb |
| `snapshot_watcher.py` | copies the encoder every N steps (train.py keeps only the latest); standard library only, no GPU use |
| `probe.py` | the go/no-go: frozen features → the 40-epoch MLP probe on MyFonts tags → ICCV mAP / NDCG, AMT, top-50 ROC/PR-AUC. `--control` probes old ViT checkpoints; `--bitmap64` reads 64px glyphs |
| `embed.py` | `loadEncoder` / `embedFont` / `embedFonts`, plus a CLI that embeds the whole cache to `embeddings/levjepa_<run>_<step>.npz/.json` |

## Results (`results/levjepaProbe/`)

The run `levjepa-d256-e200` (256 wide, 12 layers, 7.9M parameters, 128 fonts per step) was stopped at step 38,500 of
59,000. Held-out invariance had flattened and the probe had plateaued since step 25k.

Full split: 15,049 train fonts, 1,876 test fonts. All models see all 52 glyphs unless noted.

| model | top-50 ROC-AUC | mAP single-300 / full / multi | AMT |
|---|---|---|---|
| untrained encoder | 0.748 | 14.34 / 6.93 / 6.49 | 0.456 |
| LeVJEPA step 25,000 | 0.798 | 20.19 / 11.36 / 11.70 | 0.484 |
| LeVJEPA step 38,500 | 0.799 | 20.27 / 11.41 / 12.01 | 0.497 |
| LeVJEPA step 38,500, sparse training-style views | 0.800 | 20.41 / 11.57 / 12.14 | 0.495 |
| `pretrain/best`, 26 lowercase glyphs | 0.799 | 20.55 / 11.63 / 10.76 | 0.486 |
| `pretrain/best` | 0.801 | 21.05 / 12.04 / 11.08 | 0.483 |
| `weak-sigreg-proj-48px-patch4-d256-a0.5` (144 tokens/glyph) | 0.800 | 20.79 / 12.17 / 11.29 | 0.475 |
| `strong-sigreg-64px-patch8` (64px; strong SIGReg, no projector) | 0.769 | 16.90 / 8.58 / 7.29 | 0.464 |

For reference, the ICCV 2019 paper reports 26.29 / 16.77 / 14.93 for its end-to-end ResNet-50 basic classifier and
28.08 / 18.02 / 16.74 for its full model.
