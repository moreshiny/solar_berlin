# Data cleaning in the absence of

from roof.automated_data_cleaning import DataCleaning


PATH_TO_CLEAN = "data/j_to_clean"


# call the cleaning class.
cleaning = DataCleaning(
    path_to_clean=PATH_TO_CLEAN,
    input_shape=(512, 512, 3),
)

discard_list = cleaning.manual_sorting()


print("Manual sorting Done")


#list_discarded = cleaning.move_discarded_files(
#    output_folder_name="dirty",
#    delete_existing_output_path_no_warning=True,
#)

#print(list_discarded)
print("File moved.")


print("Done")
