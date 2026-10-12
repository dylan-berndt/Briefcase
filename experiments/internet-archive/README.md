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
found with an SVD projection + faiss range search, then re-scored exactly on the full 18,432-pixel vectors; the
search radius is lowered by the measured projection error so no pair above min(t) is missed (up to a 99.99%
quantile).

```bash
python experiments/internet-archive/fetch.py "Fonts Mega Pack 102.000 Fonts.zip"
python experiments/internet-archive/build_cache.py "dataset/internetArchive/fonts/Fonts Mega Pack 102.000 Fonts"
python experiments/internet-archive/dedup.py "dataset/internetArchive/cache/Fonts Mega Pack 102.000 Fonts"
```

Dedup should eventually run over all chunk caches together (`dedup.py cacheA cacheB ...`), since the zips
duplicate each other.

## Results
