#!/bin/sh

DATA_FOLDER="../data"

# set the raw data folder
RAW_DATA_FOLDER="$DATA_FOLDER/raw"

# set the raw raster folder
RAW_RASTER_FOLDER="$RAW_DATA_FOLDER/raster"

# set the overlay shape folder
RAW_OVERLAY_FOLDER="$RAW_DATA_FOLDER/overlay"

# create the raw data folder
mkdir -p $RAW_DATA_FOLDER

# download the 2013 arial image raster file zips:
wget -nc https://fbinter.stadt-berlin.de/fb/atom/DOP/dop20true_2013/Mitte.zip \
        -O $RAW_RASTER_FOLDER/Mitte.zip
wget -nc https://fbinter.stadt-berlin.de/fb/atom/DOP/dop20true_2013/Nord.zip \
        -O $RAW_RASTER_FOLDER/Nord.zip
wget -nc https://fbinter.stadt-berlin.de/fb/atom/DOP/dop20true_2013/Nordost.zip \
        -O $RAW_RASTER_FOLDER/Nordost.zip
wget -nc https://fbinter.stadt-berlin.de/fb/atom/DOP/dop20true_2013/Nordwest.zip \
        -O $RAW_RASTER_FOLDER/Nordwest.zip
wget -nc https://fbinter.stadt-berlin.de/fb/atom/DOP/dop20true_2013/Ost.zip \
        -O $RAW_RASTER_FOLDER/Ost.zip
wget -nc https://fbinter.stadt-berlin.de/fb/atom/DOP/dop20true_2013/Sued.zip \
        -O $RAW_RASTER_FOLDER/Sued.zip
wget -nc https://fbinter.stadt-berlin.de/fb/atom/DOP/dop20true_2013/Suedost.zip \
        -O $RAW_RASTER_FOLDER/Suedost.zip
wget -nc https://fbinter.stadt-berlin.de/fb/atom/DOP/dop20true_2013/Suedwest.zip \
        -O $RAW_RASTER_FOLDER/Suedwest.zip
wget -nc https://fbinter.stadt-berlin.de/fb/atom/DOP/dop20true_2013/West.zip \
        -O $RAW_RASTER_FOLDER/West.zip

# downlaod the vector representation of solar potential:
wget -nc https://fbinter.stadt-berlin.de/fb/atom/solar/ST_PV_Potenzial_2013.zip \
        -O $RAW_OVERLAY_FOLDER/ST_PV_Potenzial_2013.zip

# check the sha256sum of the downloaded files:
sha256sum -c sha256sum.txt

# create new sha256sum file from downloaded files:
# sha256sum $RAW_RASTER_FOLDER/*.zip $RAW_OVERLAY_FOLDER/*.zip > sha256sum.txt
