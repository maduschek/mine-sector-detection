TARGDIR="/home/maduschek/src/mine-sector-detection/images/"
FILES="/home/maduschek/data/mine-sectors/mapbox_mines_0.8m_RGB/images/*.jp2"

for f in $FILES
do
  echo "Processing file $f"
  /usr/bin/gdal_retile.py -v -ps 256 256 -overlap 128 -co "tiled=YES" -targetDir $TARGDIR $f
done


