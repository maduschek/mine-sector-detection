import glob
import os
from PIL import Image
import subprocess
import pdb
from osgeo import gdal
import operator

# gdal_rasterize -l LSM_Chile_sectors_class_epsg3857 -a class -ts 25729.0 13441.0 -a_nodata 0.0 -te -7663806.3641 -2396313.3772 -7633077.4844 -2380260.4069 -ot Float32 -of GTiff "C:/Dropbox/TUM SIPEO/Projekte/RS mining facilities/L-ASM Mining Dataset/ds/LSM_Chile_sectors_class_epsg3857.geojson" C:/Users/ma>


# creates mask files based on images and geojson file

# path_images = "C:/data/mine-sectors/mapbox_mines_0.8m_RGB/"
path_images = "../../ssd/mine-sector-detection/images/"
path_masks = "../../ssd/mine-sector-detection/masks/"


# pdb.set_trace()
for img, msk in zip(sorted(glob.glob(path_images + "*.jp2")), sorted(glob.glob(path_masks + "*.tif"))):

    ds_img = gdal.Open(img)
    ds_msk = gdal.Open(msk)

    if ds_img.RasterXSize != ds_msk.RasterXSize or ds_img.RasterYSize != ds_msk.RasterYSize:
        print("###################################################")
        print(img, "    ",  msk)
        print("image-size: ", ds_img.RasterXSize , "x", ds_img.RasterYSize, " mask-size: ", ds_msk.RasterXSize, "x", ds_msk.RasterYSize)
        print(" ")
        input("press")


