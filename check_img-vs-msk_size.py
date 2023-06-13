import glob
import os
from PIL import Image
import subprocess
import pdb
from osgeo import gdal
import operator

# checks if the image size equals the mask size

# path_images = "C:/data/mine-sectors/mapbox_mines_0.8m_RGB/"
path_images = "/home/maduschek/ssd/mine-sector-detection/images_trainset/"
path_masks = "/home/maduschek/ssd/mine-sector-detection/masks_trainset/"


filecount = len(glob.glob(path_images + "*.png"))


# pdb.set_trace()
i=0
for img, msk in zip(sorted(glob.glob(path_images + "*.png")), sorted(glob.glob(path_masks + "*.png"))):

    ds_img = gdal.Open(img)
    ds_msk = gdal.Open(msk)
    i += 1
    if i % 1000 == 0:
        print(str(i), " / ", str(filecount))

    if ds_img.RasterXSize != ds_msk.RasterXSize or ds_img.RasterYSize != ds_msk.RasterYSize:
        print("###################################################")
        print(img, "    ",  msk)
        print("image-size: ", ds_img.RasterXSize , "x", ds_img.RasterYSize, " mask-size: ", ds_msk.RasterXSize, "x", ds_msk.RasterYSize)
        print(" ")
        input("press")


