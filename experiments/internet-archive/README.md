# Internet Archive font collection

More pretraining data: the Internet Archive item
[`1.5_million_font_files_collection`](https://archive.org/details/1.5_million_font_files_collection) ("1.5 Million
Fonts Compilation", 2018, no stated licence), 60.24 GB over 13 zips. The zip listings hold ~1.05M `.ttf`/`.otf`
files (plus ~100k Type 1 `.pfb` and a few thousand nested zips, not used), with ~219k distinct file names, which is
where the "201k unique" figure comes from: the zips are overlapping compilations of each other.

`fetch.py` downloads zips (resumable, md5-checked) and extracts only `.ttf`/`.otf`/`.ttc` into
`dataset/internetArchive/fonts/<zip stem>/` (`--list` shows the zips). From there the collection is one more
`standard` font directory for the existing loaders:

* `utils/loaders/standard.py` caches its glyphs like DaFont's and Google's (`configs/vit.json` lists it), and
  `experiments/levjepa/build_glyph_cache.py` packs it with the rest (`configs/levjepa.json`). Both use the same
  MyFonts-style render, and font keys are `"<family> <style> <sha256[:12]>"`, so byte-identical copies collapse and
  different files that share a name stay apart.
* The near-duplicate step (`utils/dedup.py`) runs after every source is cached and loaded, over the whole compiled
  corpus: `PairedImageData` with the config key `dedupThreshold`, `build_glyph_cache.py` (writes `dedup.npz`/`dedup.json`),
  and `python -m utils.dedup --config ... | --glyphCache ...` for the table.

Dedup rule per t: each pair with s > t is visited in random order and, if both fonts are still kept, one of them chosen
at random is discarded (mean over 5 orders for the table, seed 0 for training). Similarity is the cosine of 8 glyphs
(`aegknHQR`) concatenated, each through `centerAndScaleNormalize`. Fonts that draw one placeholder box for every letter
are always dropped; fonts missing a dedup letter are kept and not compared. Pairs are found exactly (blockwise matmul,
GPU through torch when available).

```bash
python experiments/internet-archive/fetch.py "Fonts Mega Pack 102.000 Fonts.zip"
python experiments/levjepa/build_glyph_cache.py                     # every source + dedup table
python -u experiments/embedding-geometry/build_bitmap_cache.py --fontSize 32 --cacheSuffix "" --only standard   # utils cache
python -m utils.dedup --config configs/vit.json
```

## Results

**Mega Pack chunk on its own** (`Fonts Mega Pack 102.000 Fonts.zip`, Oct 2026, CPU; mean over 5 random orders; run with
the first standalone version of this pipeline, same render and normalization, but fonts missing a dedup letter were
left out of the corpus count instead of kept). 72,781 font
files, all distinct by sha256. 3,967 are not comparable (80 fail to open; the rest lack one of `aegknHQR`, mostly
symbol/non-Latin fonts) and 3,669 draw the same placeholder box for every letter (they pass the cmap check, and
nearly identical letters are how `dedup.py` catches them). **Corpus before dedup: 65,145.**

| t | pairs with s > t | discarded | % discarded | corpus after |
|---|---|---|---|---|
| none | | 0 | 0.0% | 65,145 |
| 0.90 | 3,338,719 | 39,845 | 61.2% | 25,300 |
| 0.92 | 1,244,652 | 36,800 | 56.5% | 28,345 |
| 0.94 | 417,451 | 33,178 | 50.9% | 31,967 |
| 0.95 | 258,963 | 31,174 | 47.9% | 33,971 |
| 0.96 | 171,115 | 28,998 | 44.5% | 36,147 |
| 0.97 | 117,303 | 26,495 | 40.7% | 38,650 |
| 0.98 | 82,919 | 23,715 | 36.4% | 41,430 |
| 0.99 | 50,825 | 19,485 | 29.9% | 45,660 |
| 0.995 | 31,487 | 15,340 | 23.5% | 49,805 |
| 0.999 | 11,256 | 8,167 | 12.5% | 56,978 |

What the thresholds mean, from sampled pairs (`pairs_near_t.png`): above 0.995 the pairs are the same design
under another name or foundry release (ITC Stone Serif / StoneSerif LT, ITC Benguiat Std / ITCBenguiat, Souvenir LT /
ITCSouvenir). At 0.98–0.99 there are clones and same-family optical sizes (MacroSSK / News Gothic, Arno Pro Display /
Subhead), plus some weight neighbours. At 0.96–0.97 the pairs are related but distinct fonts (Garamond 3 / URW
Garamond, Kabel / Koblenz, Concoid Bold Italic / Italic), and at 0.90–0.94 merely the same genre (Helvetica Neue
Heavy Italic / Blox). Cosine of raw pixels is high between any two text fonts (sampled median nearest neighbour
0.989, mean over all pairs 0.49), so t below ~0.99 removes distinct fonts.
