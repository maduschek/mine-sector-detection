import glob
import os.path
import platform
import numpy as np
from PIL import Image


# mine sectors
if platform.system() == "Windows":
    base_dir = "C:/data/mine-sectors/"
else:
    base_dir = "/home/maduschek/ssd/mine-sector-detection/"
    # base_dir = "/home/maduschek/data/cats_dogs/"

input_dir_train = base_dir + "images_trainset/"
target_dir_train = base_dir + "masks_trainset/"
input_dir_test = base_dir + "images_testset/"
target_dir_test = base_dir + "masks_testset/"


shape_img_list = []
shape_mask_list = []
class_instances = dict()

k = 0
for img_path, mask_path in zip(glob.glob(os.path.join(input_dir_train, "*.png")), glob.glob(os.path.join(target_dir_train, "*.png"))):

    k += 1

    # get shape of all images
    img = Image.open(img_path)
    if img.size not in shape_img_list:
        shape_img_list.append(img.size)
        print("img shape", img.size)

    # get shape of all masks
    mask = Image.open(mask_path)
    if mask.size not in shape_mask_list:
        shape_mask_list.append(mask.size)
        print("mask size", mask.size)

    # get all possible classes
    vals, counts = np.unique(np.asarray(mask), return_counts=True)

    for val, count in zip(vals, counts):
        if val in class_instances:
            class_instances[val] += count
        else:
            class_instances[val] = count

    if k % 100 == 0:
        os.system("clear")
        print(k)
        for key in class_instances.keys():
            print("class ", key, ": ", class_instances[key])

