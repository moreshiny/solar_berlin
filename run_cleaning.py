"""run file for the automated cleaning class."""
from roof.automated_data_cleaning import DataCleaning
from unet.unet_resnet101v2 import Unet

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