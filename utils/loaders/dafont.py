import math
from PIL import Image
import numpy as np
import os
import cv2
from glob import glob
from .description import *
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd


def loadDaFontDescriptions(directory):
    df = pd.read_csv(os.path.join(directory, "info.csv"), on_bad_lines="warn")

    # fileNames = ["".join(fileName.split(".")[:-1]) for fileName in df["filename"].tolist()]
    fileNames = df["base_font_name"].tolist()
    descriptors = [Description(fileNames[i], [df["category"].iloc[i], df["theme"].iloc[i]]) for i in range(len(df))]

    descriptions = dict(zip(fileNames, descriptors))

    return descriptions