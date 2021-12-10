<<<<<<< HEAD
"""run file for the automated cleaning class."""
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
PATH_CHECKPOINT = "logs/12_14_2021_19_30_44/checkpoint.ckpt"

# Loading the weights
model.load_weights(PATH_CHECKPOINT)

# Define the path of the folder containing the images to sort
PATH_TO_CLEAN = "data/bin_clean_8000/test"


# call the cleaning class.
cleaning = DataCleaning(
    path_to_clean=PATH_TO_CLEAN,
    input_shape=(512, 512, 3),
    model=model,
)

# Perform the cleaning. The proportion parameter signals the proportion of samples to study.
# create a CSV containing the paths with the highest loss.
PROPORTION = 0.3
PROPORTION_EMPTY = 0.25
cleaning.cleaning(proportion=PROPORTION, proportion_empty=PROPORTION_EMPTY)


print("Cleaning Done")


# Calling the manual sorting. If the cleaning method of the class has been called before,
# the sorting starts with the picture with the highest loss. The content is loaded from the Dataframe
# cleaning._discard_df. If a file called high_loss_elements does not exist in the folde
# the dataframe will be created from the elements of the folder.
discard_list = cleaning.manual_sorting()


print("Manual sorting Done")


# Move the marked files to be discared from the `CSV file to a folder called by default dirty.

list_discarded = cleaning.move_discarded_files(
    output_folder_name="dirty",
    delete_existing_output_path_no_warning=True,
)

print(list_discarded)
print("File moved.")


print("Done")
=======
from class_automated_data_cleaning import DataCleaning
from class_unet_resnet101v2 import Unet




output_classes = 5  # number of categorical classes.
input_shape = (512, 512, 3)  # input size

#accuracy threshild for the cleaning

# calling the model.
model = Unet(
    output_classes=output_classes,
    input_shape=input_shape,
    drop_out=False,
    drop_out_rate={"512": 0.275, "256": 0.3, "128": 0.325, "64": 0.35},
    multiclass=bool(output_classes - 1),
)


path_checkpoint = "logs/12_09_2021_22_25_55/checkpoint.ckpt"

model.load_weights(path_checkpoint)

path_to_clean = "data/selected_tiles_512_4000_1000_42_cleaned/train"

cleaning = DataCleaning(
        path_to_clean = path_to_clean,
        input_shape = (512, 512, 3),
        model = model,
)

cleaning.cleaning(proportion=0.05)


print("Done")



path_to_clean = "data/selected_tiles_512_4000_1000_42_cleaned/test"

cleaning = DataCleaning(
        path_to_clean = path_to_clean,
        input_shape = (512, 512, 3),
        model = model,
)

cleaning.cleaning(proportion=0.05)


print("Done")



>>>>>>> 7c7f149 (file to run the cleaning class)
