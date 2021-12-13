The following people contributed equally to this repository (in alphabetical order):

* [Daniel Bumke](https://github.com/moreshiny)
* [JJX](https://github.com/jjx3455)
* [Corstiaen Versteegh](https://github.com/cverst)

This repository is forked on https://github.com/jjx3455/solar_berlin and https://github.com/cverst/solar_berlin to reflect the work of all contributors.

# Estimating the potential photovoltaic production of Berlin rooftops

## Prerequisites

- Install Python 3.7 (other versions may work) and the packages listed in requirements.txt
- To run the model effectively, gpu-accelertion in tensorflow is advisable - https://www.tensorflow.org/install/gpu
- In order to convert the original raster tiles from ecw to GeoTIFF GDAL must be built with ecw support (see https://trac.osgeo.org/gdal/wiki/ECW)
- (Alternatively, skip downloading the tiles and use the pre-processed GeoTIFFs available as "converted.zip" https://drive.google.com/drive/folders/1zJGu6x-S13IBi_N0VGynjAypKGTq6JSC)
- (Alternatively 2, skip the extraction too and use the sample data available as "selected_512.zip" from https://drive.google.com/drive/folders/1zJGu6x-S13IBi_N0VGynjAypKGTq6JSC)

## Getting the original data

To download and convert the original data:
- Create a subfolder "data" in the repository
- If not using the pre-processed GeoTIFFs, download the original data:
    - Download map and vector files:
```cd conversion``` && ```sh download.sh```
    - Convert from ecw to GeoTIFF:
```sh convert_rasters.sh```
    - Clip the vector map to the same size as the raster tiles:
```python clip_shape.py```

## Extracting the data for the model

To extract the data into usable map/mask tiles and select a sample for the model:
- Adjust parameters in ```select_data.py``` for desired sample size and tile size
- Run ```python select_data.py```
- (This will take a while as it extracts all tiles of that size from the original
    data. The tiles are saved in the data/extracted folder so sampling again from
    the same tiles will be much faster.)

## Running the current model

The current model uses a pre-trained MobileNetV2 model for semantic segementation
of rooftops (pixel-by-pixel prediction of roof/no-roof).

To run the model:

- If not using the pre-extracted sample tiles:
    - modify the parameters in ```run_model.py``` to point at a different data selection folder
    - Run ```python run_model.py```
    - Inspect results in logs

## Running Tests

To run the tests:
- Additionally download "testing.zip" from https://drive.google.com/drive/folders/1zJGu6x-S13IBi_N0VGynjAypKGTq6JSC and unzip it in the data folders
- Run ```python run_tests.py```
