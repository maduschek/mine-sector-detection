import glob
import os
from PIL import Image
import subprocess

path_images = "C:/data/mine-sectors/mapbox_mines_0.8m_RGB/"
path_json = "./"
os.makedirs("./out", exist_ok=True)

for file in glob.glob(path_images + "*.jp2"):

    img = Image.open(file)
    pixelsize = str(img.size[0]) + " " + str(img.size[1])
    string = "gdal_rasterize -l out -a class -ts " + pixelsize + " -a_nodata 0.0 " \
             "-te -7825738.2085 -2675527.0926 -7821399.2129 -2672659.5097 " \
             "-ot Float32 -of JP2ECW '" + path_json + "LSM_Chile_sectors_class.geojson' " \
             "./out"

    os.system(string)


