import glob
import os
import geopandas as gpd
from osgeo import gdal

RAW_DATA_DIR = os.path.join("..", "data", "raw")
CONVERTED_DATA_DIR = os.path.join("..", "data", "converted")


shape_fn = os.path.join(RAW_DATA_DIR, "overlay", "ST_PV_Potenzial_2013.zip")
print(f"Reading {shape_fn}. This can take a while...")

shapefile = gpd.read_file(shape_fn)

raster_dir = os.path.join(CONVERTED_DATA_DIR, "raster")

out_dir = os.path.join(CONVERTED_DATA_DIR, "vector")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

raster_filenames = glob.glob(os.path.join(raster_dir, "*.tif"))

for raster_file in raster_filenames:
    print(f"Processing {raster_file}")

    # get filename without extension
    filename = os.path.basename(raster_file)
    base_filename = os.path.splitext(filename)[0]

    if os.path.exists(os.path.join(out_dir, base_filename)):
        print(f"{base_filename} already exists - skipping!")
        continue

    # get the extent of the current raster
    ds = gdal.Open(raster_file)
    x_min, x_max, y_min, y_max =\
        ds.GetGeoTransform()[0], ds.GetGeoTransform()[0]\
        + ds.RasterXSize * ds.GetGeoTransform()[1], ds.GetGeoTransform()[3]\
        + ds.RasterYSize * ds.GetGeoTransform()[5], ds.GetGeoTransform()[3]
    # ensure data is written to disk by closing ds
    del ds

    out_shapefile = shapefile.cx[x_min:x_max, y_min:y_max]

    # save the shape file matching the raster extent to the output directory
    os.makedirs(os.path.join(out_dir, base_filename))
    try:
        out_shapefile.to_file(os.path.join(
            out_dir, base_filename, base_filename + ".shp"))
    except ValueError:
        print(f"No data in shapefile for {base_filename} - skipping!")

    print(f"Finished processing {raster_file}")
