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
| `build_glyph_cache.py` | builds `dataset/levjepa/` (uint8 `[N, 52, 48, 48]` memmap + presence mask) from the sources in `configs/levjepa.json` (MyFonts, DaFont, Google, Internet Archive), all on the MyFonts render shared with `utils/loaders/standard.py`, then runs the near-duplicate step (`utils/dedup.py`: `dedup.npz`, table in `dedup.json`). The Sept 30 build had 40,112 fonts. |
| `train.py` | pretraining. Every setting that defines a run is in `configs/levjepa.json` (copy it per run; `dataset.dedupThreshold` picks the near-duplicate cut); flags are operational only. `--supervise` restarts and resumes after crashes, and resuming refuses a run whose config changed. `--rebuild` deletes the glyph cache's files and rebuilds it (all sources, then dedup) before training, once even under `--supervise`. Logs `LeVJEPA/*` to wandb |
| `snapshot_watcher.py` | copies the encoder every N steps (train.py keeps only the latest); standard library only, no GPU use |
| `probe.py` | the go/no-go: frozen features → the 40-epoch MLP probe on MyFonts tags → ICCV mAP / NDCG, AMT, top-50 ROC/PR-AUC. Settings in `configs/levjepaProbe.json`; `--control` probes old ViT checkpoints; `--bitmap64` reads 64px glyphs |
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

## Running a new pretraining run (all sources, Internet Archive included)

1. **Data.** MyFonts (`dataset/`), DaFont (`dafont/`) and Google (`google/`) as before (`initialize.sh`), plus the
   Internet Archive fonts. Download only the 7 zips that contain new file names (~36 GB download; the other 6 zips
   are re-compilations of these):
   ```bash
   python experiments/internet-archive/fetch.py \
       "Worlds Largest Font Collection 600000.zip" \
       "2012_Font_Collection.zip" \
       "ADDITIONAL SETS.zip" \
       "dafont.com_categories_2010-05-19.zip" \
       "Enormous font collection.zip" \
       "Font Collection by nonhocapito.zip" \
       "Fonts.zip"
   ```
   - It extracts only `.ttf`/`.otf`/`.ttc` into `dataset/internetArchive/fonts/<zip stem>/` and deletes each zip
     after extracting it (`--keepZip` keeps it). Peak disk use is the largest zip plus everything extracted so far.
   - Re-running is safe. Downloads resume, zips are md5-checked, and already-extracted zips are skipped.
   - `--list` shows every zip and its size.
   - `configs/levjepa.json` already lists `dataset/internetArchive` under `standard`.
2. **Config.** Copy `configs/levjepa.json` (e.g. to `configs/levjepa-ia.json`) and change:
   - `runName`: a new name, so the run can't resume an old run's checkpoint.
   - `epochs`: with ~200k+ fonts an epoch is ~6× the old one. The last run plateaued by ~25k steps, so aim for
     30–40k total steps rather than 100 epochs.
   - `dataset.dedupThreshold`: leave `null` until step 3.
3. **Build the cache and pick the dedup threshold.**
   ```bash
   python experiments/levjepa/build_glyph_cache.py --config configs/levjepa-ia.json --clean
   ```
   This renders every source, runs the near-duplicate step and prints the threshold table (also saved to
   `dataset/levjepa/dedup.json`). Put the chosen t into `dedupThreshold`; the pairs are cached, so changing t later
   is free. On the Mega Pack chunk, pairs above ~0.995 were the same design renamed and below ~0.97 distinct fonts
   (`experiments/internet-archive/README.md`).
   - Every builder output in `dataset/levjepa` is rewritten either way. `--clean` deletes them first, so a crashed
     build leaves files missing instead of a mismatched mix. It never deletes the directory itself.
   - Expect roughly 25–30 GB for `glyphs.npy` plus ~9 GB of temporary dedup vectors. The exact pair search re-reads
     that vectors file once per block of rows, so it slows down a lot if ~9 GB doesn't fit in RAM.
4. **Train.**
   ```bash
   python -u experiments/levjepa/train.py --config configs/levjepa-ia.json --supervise
   ```
   It prints the total step count at startup (needed below). Skip `--rebuild` here, since step 3 just built the
   cache. Use `--rebuild` later, once a threshold is set, whenever the sources change: it runs the same build with
   `--clean` (once, even under `--supervise`) and then trains.
5. **Monitoring** (two more terminals):
   ```bash
   python -u experiments/levjepa/snapshot_watcher.py --run <runName> --total <total steps> --every 5000
   python -u experiments/levjepa/probe.py --follow checkpoints/pretrain/<runName>-snapshots --total <total steps> --wandb
   ```
   The watcher copies the encoder every 5,000 steps; the probe scores each snapshot on the MyFonts tags
   (`results/levjepaProbe/`, wandb `LeVJEPA/probe/*`). `configs/levjepaProbe.json`'s `glyphCache` must be the same
   directory as the training config's. Each probe pass slows training for ~15 minutes.
6. **Control baseline.** The cache was re-rendered with new font keys, so re-probe the old ViT once on it rather than
   comparing against the table above:
   ```bash
   python -u experiments/levjepa/probe.py --control checkpoints/pretrain/best --glyphs all
   ```

## Finetuning tags with a LeVJEPA backbone

`finetuneTags.py` (repo root) takes a LeVJEPA run or snapshot as its backbone. Copy `configs/finetuneTags.json` and
point `model.backbone` at the folder (e.g. `checkpoints/pretrain/<runName>-snapshots/step25000`). It recognises
LeVJEPA from that folder's `config.json` (`model.type == "levjepa"`), so nothing else needs setting.

```bash
python finetuneTags.py --config configs/finetuneTags-levjepa.json
python finetuneTags.py --config configs/finetuneTags-levjepa.json --maxFonts 300 --maxEpochs 2   # smoke test
```

- **Font feature:** the encoder reads all of a font's glyphs as one token sequence and its CLS output is the font
  feature, the same embedding `probe.py` scores. During training each font gets its own random subset of letters
  (`glyphsPerFont`) and the encoder is told which letters they are. Needs `fontSize` 32 (48px glyphs).
- **Saved run:** the finetuned encoder plus the unchanged LeVJEPA `config.json`, so `embed.py`, `probe.py` and the
  site's tagger adapter (`site/tools/taggers.py`) all load it.
- **`training.autocast`** (default `true`, fp16 mixed precision): fp16 was ~5× slower per step for LeVJEPA
  pretraining on the GTX 1660 SUPER, so try `false` for LeVJEPA runs.
- **Evaluation memory:** evaluation runs batches of 32 fonts × 26 glyphs (~940 tokens per font). If that runs out of
  GPU memory, lower `batchFonts` in `finetuneTags.predict()`.
- **Tested** on CPU only, with tiny random backbones and a fake 36-font dataset: features match `embed.py` exactly
  (all 26 letters and per-font letter subsets), 2-epoch runs complete with both backbone kinds, and a saved LeVJEPA
  run reloads through `embed.loadEncoder` and the site adapter with matching outputs. Not yet run on the real
  dataset or a real checkpoint; speed and memory on the GPU are unmeasured.
