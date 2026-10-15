# Font Search

This is the code behind [Font Search](https://font-search.com), a project of mine I've wanted to use for years: a plain-text font searching tool with free fonts.
It's not exactly as I envisioned it (still relies on a set of tags for matching), but it functions about as good as I could achieve within a year's worth of work.
Either way, I've produced something good enough to put my name on. Enjoy!

I got inspired to work on this project by watching a great video by Tom7 on experimenting with uppercasing and lowercasing letters with neural networks, and all the nonsense you can do with a trained model: [watch it here](https://www.youtube.com/watch?v=HLRdruqQfRk). I recommend you watch every Tom7 video except the ones you don't like probably.

I can only hope my contributions to derp learning are worth their weight in GPU-hours.

**How it works, how it was made, and who made it: [the full about page](https://font-search.com/about).**

The website is in [`site/`](site/README.md), and the research behind it (training, evaluation, results) is in [`RESEARCH.md`](RESEARCH.md).

## License

The source code in this repository is licensed under the MIT License (see [`LICENSE`](LICENSE)). This does not cover:

- the font specimen images, font names and links in `site/backend/data/`. These belong to the font authors; see each font's own license.
- the trained model weights and tag predictions, which were trained on the MyFonts dataset from Chen et al., ICCV 2019, under that dataset's terms.
