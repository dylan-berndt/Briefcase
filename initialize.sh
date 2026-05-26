# Used when starting up training in an empty repo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Get the Google dataset
git clone https://github.com/google/fonts

# Get the MyFonts dataset
wget https://docs.google.com/uc?export=download&confirm=t&id=10GRqLu6-1JPXI8rcq23S4-4AhB6On-L6
tar -xf dataset.tar.xz --strip-components=1 -C ./dataset

# Get the DaFont (public domain) dataset
wget https://github.com/duskvirkus/dafonts-free/releases/download/v1.0.0/dafonts-free-v1.zip
unzip -j dafonts-free-v1.zip -d ./dafont