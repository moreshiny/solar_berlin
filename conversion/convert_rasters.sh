#!/bin/sh

DATA_FOLDER="../data"

# set the raw data folder that contains the raster zip files
RAW_RASTER_FOLDER="$DATA_FOLDER/raw/raster"

#set the temporary folder to extract zip files to
TEMPORARY_FOLDER="$RAW_RASTER_FOLDER/temp"
mkdir -pv $TEMPORARY_FOLDER

# set the output folder
CONVERTED_FOLDER="$DATA_FOLDER/converted/raster"
mkdir -pv $CONVERTED_FOLDER

# for each file in the raw folder
for zipfile in $RAW_RASTER_FOLDER/*.zip
do
    region=$(basename $zipfile .zip)
    echo "Converting $region"

    # get the list of files in the zip file
    unzip -l $zipfile | awk '{print $4}' | grep ".ecw$" > $TEMPORARY_FOLDER/filelist.txt

    for raster_file in $(cat $TEMPORARY_FOLDER/filelist.txt)
    do
        # remove the extension from the raster file name
        output_filename=$(basename $raster_file .ecw)

        #if the output file doesn't exist
        if [ ! -f $CONVERTED_FOLDER/$region-$output_filename.tif ]
        then
            echo "Converting $raster_file"
            unzip -n $zipfile -d $TEMPORARY_FOLDER $raster_file
            unzip -n $zipfile -d $TEMPORARY_FOLDER $output_filename.eww

            # convert the ecw to tif - this requires that gdal has been built
            # with ecw support
            gdal_translate -of GTiff \
                $TEMPORARY_FOLDER/$raster_file $CONVERTED_FOLDER/$region-$output_filename.tif \
                -co COMPRESS=DEFLATE -co NUM_THREADS=ALL_CPUS

            # remove the temporary raster file
            rm $TEMPORARY_FOLDER/$raster_file
            rm $TEMPORARY_FOLDER/$output_filename.eww
            echo "...done $raster_file"
        else
            echo "Skipping $raster_file as output file already exists"
        fi
    done
    rm $TEMPORARY_FOLDER/filelist.txt
done

# remove the temp folder
rmdir $TEMPORARY_FOLDER
