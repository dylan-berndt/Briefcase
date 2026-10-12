# Experiments

Research scripts behind the font search, in roughly the order they were done. Each folder is self-contained and runs
from the repo root. Datasets, checkpoints, caches and embeddings are gitignored; each folder says what it needs.
`CLAUDE.md` holds the running notes, including numbers and the reasoning behind each turn.

| folder | question | outcome |
|---|---|---|
| `pretrain-analysis/` | Which pretraining objective transfers best (LogME / TransRate / H-Alpha, `estimation.py`), and does the UNet's internal representation track style (`styleTesting.py`)? | Results tables in the top-level README; activation CSVs and plots kept here. |
| `unet-visualization/` | Toys for the pretrained UNet: draw a glyph and see it uppercased (`draw.py`), gradient-ascent glyph hallucination (`hallucinate.py`, outputs in `hallucinations/`). | Exploratory. |
| `choice-searches/` | Can a user find a font by picking from grids, no text (`GridFeedbackSearch`)? Also the earlier preference-walk (`meander.py`, `trainMeander.py`, `testMeander.py`) and cluster-narrowing UIs. | Works on a ~3.7k-font corpus, stalls on the full 39k: top-m gets trapped, UCB / GP-UCB-PE escape more often but slowly. `estimateSearchQuality.py` holds the 30 trial queries reused later. |
| `text-searches/` | How much does text narrow the corpus? Query generation (`generate.py`, Phi-4; `generateV2.py`, Gemma), the `search.py` pygame UI, text-to-embedding MLP. | Median target rank in the top 2.2% but R@1 ≈ 0: text only gets the neighbourhood. |
| `diffusion-searches/` | Diffusion from text to visual embedding, then interactive "decision point" search over a cluster tree. | Diffusion sampling was replaced by a supervised navigation classifier. Beam search and a calibrated 6-level tree reach 0.59 leaf success with a perfect oracle, but a real human trial was near chance at depth 0–1. |
| `embedding-geometry/` | Does better embedding geometry (SIGReg variants, projector, dedup, ABTT) fix retrieval? | Geometry proxies moved up to 10x with no change in recall. The "fonts are not differentiable" verdict here was later overturned by `critical-review/`. |
| `critical-review/` | Why is text→font retrieval stuck? A from-scratch audit. | The benchmark is label-limited; the backbone is close to published tag-retrieval numbers. Found the rendering confound and the case-mixing bug. Built a canonical tag vocabulary and query parser; a blind human trial accepted 83% of its top-8 results (`e24`, `build_page_v2.py`). |
| `levjepa/` | Does LeVJEPA-style pretraining (global/local glyph views, SIGReg, no negatives) give better tag features? | As a frozen probe it ties `pretrain/best` (slightly better on multi-tag, slightly worse on single-tag). Several frozen encoders all land at ~20–21 / 11–12 / 11–12 mAP. |
| `internet-archive/` | More pretraining fonts: the Internet Archive "1.5 Million Fonts" item (~1.05M files, ~219k distinct names). How many survive near-duplicate removal? | `fetch.py` adds it as one more `standard` font directory. Dedup over the whole compiled corpus lives in `utils/dedup.py`. On the 72,781-file Mega Pack chunk alone, t=0.999 discards 12.5%, t=0.99 30% and t=0.95 48%. Above ~0.995 the pairs are the same design renamed; below ~0.97 they are distinct fonts. |

`finetuneTags.py` (repo root) finetunes the backbone end to end on MyFonts tags. It is the only change so far that
clearly beat the frozen probes: 21.79 / 12.59 / 12.52 mAP (single-300 / full / multi).
