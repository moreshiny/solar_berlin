#!/bin/sh

# set the raw data folder that contains the raster zip files
RAW_RASTER_FOLDER="data/raw/raster"

#set the temporary folder to extract zip files to
TEMPORARY_FOLDER="$RAW_RASTER_FOLDER/temp"
mkdir -pv $TEMPORARY_FOLDER

# set the output folder
CONVERTED_FOLDER="data/converted/raster"
mkdir -pv $CONVERTED_FOLDER

# for each file in the raw folder
for zipfile in $RAW_RASTER_FOLDER/*.zip
do
    # extract the zip file
    echo "Extracting" $zipfile
    unzip $zipfile -d $TEMPORARY_FOLDER
    echo "..done. $zipfile"
    region=$(basename $zipfile .zip)
    # for each raster file in the temp folder
    for raster_file in $TEMPORARY_FOLDER/*.ecw
    do
        echo "Converting $raster_file"
        # remove the extension from the raster file name
        output_filename=$(basename $raster_file .ecw)

        #if the output file doesn't exist
        if [ ! -f $CONVERTED_FOLDER/$region-$output_filename.tif ]
        then
            # convert the ecw to tif - this requires that gdal has been built
            # with ecw support
            gdal_translate -of GTiff \
                $raster_file $CONVERTED_FOLDER/$region-$output_filename.tif \
                -co COMPRESS=DEFLATE -co NUM_THREADS=ALL_CPUS
        else
            echo "Skipping $raster_file as output file already exists"
        fi
        # remove the temporary raster file
        rm $raster_file
        rm $TEMPORARY_FOLDER/$output_filename.eww
        echo "...done $raster_file"
    done
done

# remove the temp folder
rmdir $TEMPORARY_FOLDER
