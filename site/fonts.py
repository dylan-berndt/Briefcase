import os
import shutil
from glob import glob
from PIL import ImageFont

# Just puts all the fonts in place for the site's backend to serve users


folders = ["data", "windows", "google"]
cwd = ".." if os.getcwd().endswith("site") else ""

for folder in folders:
    paths = glob(os.path.join(cwd, folder, "*.ttf")) + glob(os.path.join(cwd, folder, ".otf"))
    for path in paths:
        font = ImageFont.truetype(path)
        fontName, fontStyle = font.getname()

        newPath = os.path.join(cwd, "site", "backend", "fonts", f"{fontName} {fontStyle}")
        os.makedirs(newPath, exist_ok=True)

        descriptionFile = open(os.path.join(newPath, "descriptions.txt"), "w+")
        descriptionFile.close()

        shutil.copyfile(path, os.path.join(newPath, os.path.basename(path)))


