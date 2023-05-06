TARGDIR="/home/maduschek/ssd/mine-sector-detection/masks_trainset/"
FILES="/home/maduschek/ssd/mine-sector-detection/masks/*.tif"

mkdir -p $TARGDIR

for f in $FILES
do
  echo "Processing file $f"
  /usr/bin/gdal_retile.py -v -ps 256 256 -overlap 128 -of PNG -targetDir $TARGDIR $f
done
