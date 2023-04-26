import glob
import os
from PIL import Image
import subprocess
import pdb
from osgeo import gdal

# gdal_rasterize -l LSM_Chile_sectors_class_epsg3857 -a class -ts 25729.0 13441.0 -a_nodata 0.0 -te -7663806.3641 -2396313.3772 -7633077.4844 -2380260.4069 -ot Float32 -of GTiff "C:/Dropbox/TUM SIPEO/Projekte/RS mining facilities/L-ASM Mining Dataset/ds/LSM_Chile_sectors_class_epsg3857.geojson" C:/Users/matthias/AppData/Local/Temp/processing_KFVIRc/2f32c9c1db924811af36352bf5f8fdf4/OUTPUT.tif


Image.MAX_IMAGE_PIXELS = 10000000000

path_images = "C:/data/mine-sectors/mapbox_mines_0.8m_RGB/"
path_images = "../../data/mine-sectors/mapbox_mines_0.8m_RGB/images/"
path_json = "./"
os.makedirs("./out", exist_ok=True)

# pdb.set_trace()
for file in sorted(glob.glob(path_images + "*.jp2")):

    # pdb.set_trace()
    ds = gdal.Open(file, gdal.GA_ReadOnly)
    geoTransform = ds.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * ds.RasterXSize
    miny = maxy + geoTransform[5] * ds.RasterYSize
    data = None
    rb = (ds.GetRasterBand(1)).ReadAsArray()
    pixelsize = str(rb.shape[1]) + " " + str(rb.shape[0])

    print(file)
    print([minx, miny, maxx, maxy])
    print(pixelsize)

    # "-te -7825738.2085 -2675527.0926 -7821399.2129 -2672659.5097 " \

    string = "gdal_rasterize -l LSM_Chile_sectors_class_epsg3857 -a class -ts " + pixelsize + " -a_nodata 0 " \
             "-te " + str(minx) + " " + str(miny) + " " + str(maxx) + " " + str(maxy) + " " \
             "-ot Byte -of GTiff '" + path_json + "LSM_Chile_sectors_class_epsg3857.geojson' " \
             "./out/mask_" + os.path.basename(file)[:-4] + ".tif"

    print(string)
    # input("press enter")

    os.system(string)


