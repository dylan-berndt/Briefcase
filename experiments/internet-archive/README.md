# Internet Archive font collection

More pretraining data: the Internet Archive item
[`1.5_million_font_files_collection`](https://archive.org/details/1.5_million_font_files_collection) ("1.5 Million
Fonts Compilation", 2018, no stated licence), 60.24 GB over 13 zips. The zip listings hold ~1.05M `.ttf`/`.otf`
files (plus ~100k Type 1 `.pfb` and a few thousand nested zips, not used), with ~219k distinct file names, which is
where the "201k unique" figure comes from: the zips are overlapping compilations of each other.

| script | what it does |
|---|---|
| `fetch.py` | Downloads zips (resumable, md5-checked) and extracts only `.ttf`/`.otf`/`.ttc` into `dataset/internetArchive/fonts/<zip stem>/`. `--list` shows the zips. |
| `build_cache.py` | One render per unique file (sha256): the 52 training glyphs on the standardized MyFonts pipeline (same as `levjepa/build_glyph_cache.py`, same `glyphs.npy`/`present.npy` format), plus 8 dedup glyphs (`aegknHQR`) passed through `centerAndScaleNormalize`. Resumable by shard; fonts that crash FreeType are isolated per process. |
| `dedup.py` | Pairwise cosine similarity of the 8 concatenated dedup glyphs over one or more caches (merged by sha256), and the discard count per threshold t. |

Dedup rule per t: byte-identical copies collapse to one file, then each pair with s > t is visited in random
order and, if both fonts are still kept, one of them chosen at random is discarded (mean over 5 orders). Pairs are
found exactly (blockwise matmul over the upper triangle, GPU through torch when available).

```bash
python experiments/internet-archive/fetch.py "Fonts Mega Pack 102.000 Fonts.zip"
python experiments/internet-archive/build_cache.py "dataset/internetArchive/fonts/Fonts Mega Pack 102.000 Fonts"
python experiments/internet-archive/dedup.py "dataset/internetArchive/cache/Fonts Mega Pack 102.000 Fonts"
```

Dedup should eventually run over all chunk caches together (`dedup.py cacheA cacheB ...`), since the zips
duplicate each other.

## Results

**Mega Pack chunk** (`Fonts Mega Pack 102.000 Fonts.zip`, Oct 2026, CPU; mean over 5 random orders). 72,781 font
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
