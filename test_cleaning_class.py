"""run file for the automated cleaning class."""
import glob
import json
import os
from class_automated_data_cleaning import DataCleaning
from class_unet_resnet101v2 import Unet

# Define the parameters of the model to be used.
OUTPUT_CLASSES = 5  # number of categorical classes.
INPUT_SHAPE = (512, 512, 3)  # input size


# calling the model. For inference, drop out must be deactivated.
# The structure of the model must be identical to the loaded model.
model = Unet(
    output_classes=OUTPUT_CLASSES,
    input_shape=INPUT_SHAPE,
    drop_out=False,
    multiclass=bool(OUTPUT_CLASSES - 1),
)

# Path to check point. This is a directory. Lead to an error when calling it.
PATH_CHECKPOINT = "logs/best_model_15_12/checkpoint.ckpt"

# Loading the weights
model.load_weights(PATH_CHECKPOINT)

# Define the path of the folder containing the images to sort
PATH_TO_CLEAN = "data/selected/selected_tiles_512_1000_200_42_cleaning/train"


# call the cleaning class.
cleaning = DataCleaning(
    path_to_clean=PATH_TO_CLEAN,
    input_shape=(512, 512, 3),
    model=model,
)

# Perform the cleaning. The proportion parameter signals the proportion of samples to study.
# create a CSV containing the paths with the highest loss.
PROPORTION = 0.2
PROPORTION_EMPTY = 0.1
cleaning.cleaning(proportion=PROPORTION, proportion_empty=PROPORTION_EMPTY)


print("Cleaning Done")


# Calling the manual sorting. If the cleaning method of the class has been called before,
# the sorting starts with the picture with the highest loss. The content is loaded from the Dataframe
# cleaning._discard_df. If a file called high_loss_elements does not exist in the folde
# the dataframe will be created from the elements of the folder.
# discard_list = cleaning.manual_sorting()


# print("Manual sorting Done")

# load the coco json and remove any image that is not in the folder.



# Move the marked files to be discared from the `CSV file to a folder called by default dirty.

list_discarded = cleaning.move_discarded_files(
        output_folder_name = "dirty",
        delete_existing_output_path_no_warning=True,
    )

# print(list_discarded)
print("File moved.")


# create a coco json file for this tile
coco_json = {
    "info": {
        "description": "",
        "url": "",
        "version": "",
        "year": 2021,
        "contributor": "",
        "date_created": "",
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 0,
            "name": "pv1",
            "supercategory": "roof",
        },
        {
            "id": 1,
            "name": "pv2",
            "supercategory": "roof",
        },
        {
            "id": 2,
            "name": "pv3",
            "supercategory": "roof",
        },
        {
            "id": 3,
            "name": "pv4",
            "supercategory": "roof",
        },
    ],
}

ids = []
dirty_files = glob.glob(os.path.join(PATH_TO_CLEAN, "dirty", "*_map.png"))
dirty_files.sort()
dirty_map_fns = []
for image_fn in dirty_files:
    if "_map.png" in image_fn:
        dirty_map_fns.append(os.path.basename(image_fn))


with open(os.path.join(PATH_TO_CLEAN, "coco.json"), "r") as f:
    all_coco = json.load(f)
print(dirty_map_fns)
for image in all_coco["images"]:
    print(image["file_name"])
    if image["file_name"] not in dirty_map_fns:
        coco_json["images"].append(image)
        ids.append(image["id"])
ids = list(set(ids))
for annotation in all_coco["annotations"]:
    if annotation["image_id"] in ids:
        annotation["id"] = len(coco_json["annotations"])
        coco_json["annotations"].append(annotation)

with open(os.path.join(PATH_TO_CLEAN, "coco_clean.json"), "w") as f:
    json.dump(coco_json, f)


print("Done")
