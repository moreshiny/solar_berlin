# Data cleaning in the absence of

from class_automated_data_cleaning import DataCleaning


<<<<<<< HEAD
<<<<<<< HEAD
PATH_TO_CLEAN = "data/j_to_clean"
=======
PATH_TO_CLEAN = "data/bin_clean_4000/test"
>>>>>>> 49ab123 (test cleaning file to run the test cleaning class without automated cleaning)
=======
PATH_TO_CLEAN = "data/j_to_clean"
>>>>>>> a90ca2b (updated cleaning class: calculation of remaining updated)


# call the cleaning class.
cleaning = DataCleaning(
    path_to_clean=PATH_TO_CLEAN,
    input_shape=(512, 512, 3),
)

discard_list = cleaning.manual_sorting()


print("Manual sorting Done")


<<<<<<< HEAD
<<<<<<< HEAD
list_discarded = cleaning.move_discarded_files(
    output_folder_name="dirty",
    delete_existing_output_path_no_warning=True,
<<<<<<< HEAD
 )
=======
)
>>>>>>> 49ab123 (test cleaning file to run the test cleaning class without automated cleaning)
=======
#list_discarded = cleaning.move_discarded_files(
#    output_folder_name="dirty",
#    delete_existing_output_path_no_warning=True,
#)
>>>>>>> a90ca2b (updated cleaning class: calculation of remaining updated)
=======
list_discarded = cleaning.move_discarded_files(
    output_folder_name="dirty",
    delete_existing_output_path_no_warning=True,
 )
>>>>>>> ba7d0c9 (script added to calculate the metrics from a folder containing mask and predictions)

print(list_discarded)
print("File moved.")


print("Done")
