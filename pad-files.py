import glob
import os
import time
from datetime import datetime
import pdb
import platform
# from IPython.display import display
from tensorflow.keras.preprocessing.image import load_img
from PIL import ImageOps, Image
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
from imgrender import render


# mine sectors
if platform.system() == "Windows":
    base_dir = "C:/data/"
else:
    base_dir = "/home/maduschek/ssd/mine-sector-detection/"

datasets = [base_dir + "images_trainset/",
            base_dir + "masks_trainset/",
            base_dir + "images_testset/",
            base_dir + "masks_testset/"]


for dataset in datasets:
    for img_path in glob.glob(os.path.join(dataset, "*.png")):
        img = Image.open(img_path)
        print(img_path)

        if img.size != (256, 256):
            print("file: ", img_path, ', size: ', str(img.size))
            width, height = img.size
            right = 256 - width
            bottom = 256 - height
            padded_img = Image.new(img.mode, (width + right, height + bottom))
            padded_img.paste(img, (0, 0))
            padded_img.save(img_path)
