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
