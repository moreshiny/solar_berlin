The following people contributed equally to this repository (in alphabetical order):

* [Daniel Bumke](https://github.com/moreshiny)
* [JJX](https://github.com/jjx3455)
* [Corstiaen Versteegh](https://github.com/cverst)

This repository is forked on https://github.com/jjx3455/solar_berlin and https://github.com/cverst/solar_berlin to reflect the work of all contributors.

# Estimating the potential photovoltaic production of Berlin rooftops

## Running the current model

The current model uses a pre-trained MobileNetV2 model for semantic segementation
of rooftops (pixel-by-pixel prediction of roof/no-roof).

To run the model:

- Install Python 3.7 (other versions may work) and the packages listed in requirements.txt
- Create a subfolder "data" in the repository
- Download "curated_1.zip" from https://drive.google.com/drive/folders/1zJGu6x-S13IBi_N0VGynjAypKGTq6JSC and unzip it in the data folders
- Run ```python run_model.py```
- Inspect results in logs

To run the tests:

- Additionally download "map.zip" from https://drive.google.com/drive/folders/1zJGu6x-S13IBi_N0VGynjAypKGTq6JSC and unzip it in the data folders
- Run ```python run_tests.py```
