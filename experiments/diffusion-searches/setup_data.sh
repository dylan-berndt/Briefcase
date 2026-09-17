#!/usr/bin/env bash
# Sets up a CPU-only venv and downloads the font sources needed to compute
# visual embeddings with the pretrained font ViT. Deliberately skips the
# MyFonts ("dataset") corpus to keep disk/time bounded on this machine --
# see experiments/README.md for the tradeoff.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy opencv-python-headless pillow fonttools scipy \
    scikit-learn umap-learn matplotlib spacy beautifulsoup4 pandas \
    transformers pygtrie sentence-transformers gdown

# Google Fonts (shallow clone -- full history is enormous and unnecessary)
if [ ! -d google/fonts ]; then
    mkdir -p google
    git clone --depth 1 https://github.com/google/fonts google/fonts
fi

# DaFont (public domain subset)
if [ ! -d dafont/fonts ]; then
    mkdir -p dafont
    wget -O dafont/dafonts-free-v1.zip \
        https://github.com/duskvirkus/dafonts-free/releases/download/v1.0.0/dafonts-free-v1.zip
    unzip -j dafont/dafonts-free-v1.zip -d dafont
    mkdir -p dafont/fonts
    mv dafont/*.ttf dafont/fonts/ 2>/dev/null || true
    mv dafont/*.otf dafont/fonts/ 2>/dev/null || true
    rm dafont/dafonts-free-v1.zip
fi

# MyFonts / Rochester dataset (~36k fonts, pre-rendered glyph images).
# Size is unknown ahead of time, so we bail out rather than fill the disk.
MIN_FREE_GB=5
if [ ! -d dataset/fontimage ]; then
    mkdir -p dataset
    avail=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
    if [ "$avail" -lt 15 ]; then
        echo "WARNING: only ${avail}G free before MyFonts download, skipping it." >&2
    else
        gdown 10GRqLu6-1JPXI8rcq23S4-4AhB6On-L6 -O dataset.tar.gz
        avail=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
        if [ "$avail" -lt "$MIN_FREE_GB" ]; then
            echo "WARNING: only ${avail}G free after downloading dataset.tar.gz, not extracting." >&2
            rm -f dataset.tar.gz
        else
            tar -xf dataset.tar.gz --strip-components=1 -C ./dataset
            rm dataset.tar.gz
        fi
    fi
fi

echo "Data setup complete."
