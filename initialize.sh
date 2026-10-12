# Used when starting up training in an empty repo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Get the Google dataset
git clone https://github.com/google/fonts
mkdir google
mv fonts google/fonts

# Get the MyFonts dataset
pip install gdown
gdown 10GRqLu6-1JPXI8rcq23S4-4AhB6On-L6
mkdir dataset
tar -xf dataset.tar.gz --strip-components=1 -C ./dataset
rm dataset.tar.gz

# Get the DaFont (public domain) dataset
wget https://github.com/duskvirkus/dafonts-free/releases/download/v1.0.0/dafonts-free-v1.zip
unzip -j dafonts-free-v1.zip -d ./dafont
mkdir ./dafont/fonts
mv ./dafont/*.ttf ./dafont/fonts
mv ./dafont/*.otf ./dafont/fonts
rm dafonts-free-v1.zip

# Get the Internet Archive "1.5 Million Fonts Compilation" (experiments/internet-archive/).
# Only the 7 zips that contain new file names (~36 GB download; together they hold all ~219k distinct names);
# the other 6 are re-compilations of these. Extracts fonts to dataset/internetArchive/fonts/, deletes each zip.
python3 experiments/internet-archive/fetch.py \
    "Worlds Largest Font Collection 600000.zip" \
    "2012_Font_Collection.zip" \
    "ADDITIONAL SETS.zip" \
    "dafont.com_categories_2010-05-19.zip" \
    "Enormous font collection.zip" \
    "Font Collection by nonhocapito.zip" \
    "Fonts.zip"

# Cache every source's glyphs, then find near-duplicates across the whole corpus (prints the threshold table).
# LeVJEPA cache (configs/levjepa.json):
python3 experiments/levjepa/build_glyph_cache.py
# utils bitmap caches (configs/vit.json), rendered in parallel, then the dedup table for that corpus:
python3 -u experiments/embedding-geometry/build_bitmap_cache.py --fontSize 32 --cacheSuffix "" --only all
python3 -m utils.dedup --config configs/vit.json
