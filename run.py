import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
import glob


# Root directory of the project
ROOT_DIR = os.path.abspath("./Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class MineSectorConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "mining-sectors"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 9  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = MineSectorConfig()
config.display()


class MineSectDataset(utils.Dataset):

    def __init__(self, path):
        self.path = path

        
    def load_mine_sectors(self):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """

        # Add classes
        self.add_class("mine_sector", 3, "lh")
        self.add_class("mine_sector", 4, "mf")
        self.add_class("mine_sector", 5, "op")
        self.add_class("mine_sector", 6, "pp")
        self.add_class("mine_sector", 7, "sy")
        self.add_class("mine_sector", 8, "tsf")
        self.add_class("mine_sector", 9, "wr")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        df = create_df()
        print('Total Images: ', len(df))
        mine_ids = np.array([])

        # split mines into train, valid and test

        # get unique mine ids
        for patch in df['id'].values:
            mine_id = int(patch.split(".")[0])
            if mine_id not in mine_ids:
                self.mine_ids = np.append(self.mine_ids, mine_id)

    
    def load_image(self, filename):
        # loads the image from a file, but

        pdb.set_trace()
        fpath = os.path.join(self.path, filename)
        image = cv2.imread(image_id)
        return image


    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """

        return mask.astype(np.bool), class_ids.astype(np.int32)


    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)






IMG_PATH = 'dataset/images/'
MASK_PATH = 'dataset/masks/'

batch_size = 32
num_classes = 10

# create dataframe to get the image name/index in order
def create_df():
    name = []
    for dir, subdir, filenames in os.walk(IMG_PATH):
        for filename in filenames:
            name.append(filename[:-4])

    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))



if __name__ == "__main__":

    pdb.set_trace()
    mine_sect_ds = MineSectDataset(IMG_PATH)
    sectors = mine_sect_ds.load_mine_sectors()
    mine_sect_ds.load_image(sectors[0])



    MID_trainval, MID_test = train_test_split(mine_ids, test_size=0.15, random_state=42)
    MID_train, MID_val = train_test_split(MID_trainval, test_size=0.25, random_state=42)

    X_train = np.array([])
    for id in MID_train:
        X_train = np.append(X_train, np.transpose([os.path.basename(x) for x in glob.glob(os.path.join(IMG_PATH, str(int(id)) + "*.tif"))]))

    X_val = np.array([])
    for id in MID_val:
        X_val = np.append(X_val, np.transpose([os.path.basename(x) for x in glob.glob(os.path.join(IMG_PATH, str(int(id)) + "*.tif"))]))

    X_test = np.array([])
    for id in MID_test:
        X_test = np.append(X_test, np.transpose([os.path.basename(x) for x in glob.glob(os.path.join(IMG_PATH, str(int(id)) + "*.tif"))]))


    print('Train Size: ', len(X_train))
    print('Validation Size: ', len(X_val))
    print('Test Size: ', len(X_test))
