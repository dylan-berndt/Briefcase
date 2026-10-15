# Font Search

Plain-text search for free fonts, live at [font-search.com](https://font-search.com): describe the font you want and it ranks about 12,800 fonts from Google Fonts and DaFont. It's not exactly what I first imagined (it still matches on a fixed set of tags), but it works about as well as I could get it in a year, and it started from a [Tom7 video](https://www.youtube.com/watch?v=HLRdruqQfRk) about uppercasing and lowercasing letters with neural networks.

**How it works.** A vision transformer is pretrained on glyph images from thousands of font files, then fine-tuned (LoRA) to predict about 1,800 style tags (bold, elegant, retro, ...) from the MyFonts tag dataset ([Chen et al., ICCV 2019](https://arxiv.org/abs/1909.02072)). Every font's tag probabilities are computed offline. A query is mapped onto tags (an alias table, stemming, WordNet and Datamuse synonyms) and fonts are ranked by those probabilities with a semantic multinomial (Turnbull et al., 2008). The server is Flask and numpy over a precomputed bundle, with no torch at serve time, behind a React frontend. Fonts are never served, only pre-rendered specimen images.

**Layout.** `utils/` models, data loading and search. `configs/` JSON configs for every run. `experiments/` research code. `site/` the website: backend, frontend and build tools ([`site/README.md`](site/README.md)).

The full write-up, from how it was built to who built it, is on [the about page](https://font-search.com/about).

## License

The source code in this repository is licensed under the MIT License (see [`LICENSE`](LICENSE)). This does not cover:

- the font specimen images, font names and links in `site/backend/data/`. These belong to the font authors; see each font's own license.
- the trained model weights and tag predictions, which were trained on the MyFonts dataset from Chen et al., ICCV 2019, under that dataset's terms.
