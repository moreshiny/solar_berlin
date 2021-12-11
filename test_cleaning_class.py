from class_automated_data_cleaning import DataCleaning
from class_unet_resnet101v2 import Unet

# Define the parameters of the model to be used.
output_classes = 5  # number of categorical classes.
input_shape = (512, 512, 3)  # input size


# calling the model. For inference, drop out must be deactivated.
# The structure of the model must be identical to the loaded model.
model = Unet(
    output_classes=output_classes,
    input_shape=input_shape,
    drop_out=False,
    multiclass=bool(output_classes - 1),
)

# Path to check point. This is a directory. Lead to an error when calling it.
PATH_CHECKPOINT = "Logs/12_09_2021_22_25_55/checkpoint.ckpt"

# Loading the weights
model.load_weights(PATH_CHECKPOINT)

# Define the path of the folder containing the images to sort
PATH_TO_CLEAN = "data/selected_512_multiclass/selected_tiles_512_100_20_42/train"


# call the cleaning class.
cleaning = DataCleaning(
    path_to_clean=PATH_TO_CLEAN,
    input_shape=(512, 512, 3),
    model=model,
)

# Perform the cleaning. The proportion parameter signals the proportion of samples to study.
# These samples are copied in a folder called dirty by default.
# if the folder already exists, an exception will be raised.
# To delete, use the parameters of the _move_bad_files method of the class.
PROPORTION = 0.1
cleaning.cleaning(proportion=PROPORTION, out_folder_name="dirty")


print("Done")


# Calling the manual sorting. If the cleaning method of the class has been called before,
#the sorting starts with the picture with the highest loss. The content is loaded from the Dataframe
# cleaning._discard_df. if not, define the classe, and
# specifiy here te path to clean by setting class_called = False, and
#path_to_clean = "your_path". Return the list of paths manually discarded.
discard_list = cleaning.manual_sorting(class_called=True)
#Stand-alone usage
#discard_list = cleaning.manual_sorting(class_called=False, path_to_clean = PATH_TO_CLEAN)


#Print the list of the paths manually discared into a folder to save it.
with open(
    PATH_TO_CLEAN + "/discarded.txt", "a", encoding="utf-8"
) as discarded_elements:
    for path in discard_list:
        discarded_elements.write(path)
        discarded_elements.write("\n")


print("Done")
