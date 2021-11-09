import os
import random
import shutil


class OutputPathExistsError(Exception):
    """Raised when the output path exists."""
    pass

def select_random_map_images(train_size: int, test_size: int,
                             input_path: str) -> list:
    """Selects a random subset of map images from the input path and returns
    them as a List of two Lists (one each for train and test) of tuples pairs
    of map and mask images.

    Args:
        train_size (int): number of training images to select
        test_size (int): number of test images to select
        input_path (str): location of input files

    Returns:
        list: List of lists of tuples pairs of map and mask images
    """
    # get all files in input directory
    files = os.listdir(input_path)
    files_map = [file for file in files if "map" in file]
    files_mask = [file for file in files if "mask" in file]

    # sort files by name
    files_map.sort()
    files_mask.sort()

    files_zipped = list(zip(files_map, files_mask))

    # shuffle the file pairs
    random.shuffle(files_zipped)

    # select train and test from (shuffled) front
    files_train = files_zipped[:train_size]
    files_test = files_zipped[train_size:train_size+test_size]

    # for now just return these as a list
    # TODO define clearer file type for loaded data
    return [files_train, files_test]


def copy_image_files(image_files: list, input_path: str,
                     output_path: str, delete_existing_output_path_no_warning=False):
    """Copy image files from the input path to the output path.

    Args:
        image_files (list): Filenames as returned by select_random_map_images
        input_path (str): original file location
        output_path (str): file location to copy to
        delete_existing_output_path_no_warning (bool, optional): Delete output
        path first if it exists, without warning. Defaults to False.
    """

    # if we have been asked to, delete existing out path without warning
    # TODO is this safe?
    if delete_existing_output_path_no_warning and os.path.exists(output_path):
        shutil.rmtree(output_path)

    # create output path if it doesn't exist or end if it does
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise OutputPathExistsError("At least one of the output directory already exists."
               "\nSet delete_existing=True to remove it.")

    # get file names into a dict for easier processing
    files = {}
    files["train"] = image_files[0]
    files["test"] = image_files[1]

    for subfolder in ["train", "test"]:
        output_path_subfolder = os.path.join(output_path, subfolder)

        # create output folder if it doesn't exist
        if not os.path.exists(output_path_subfolder):
            os.makedirs(output_path_subfolder)

        # copy files to output folder
        for file_tuple in files[subfolder]:
            for file_path in file_tuple:
                full_path_in = os.path.join(input_path, file_path)
                full_path_out = os.path.join(output_path_subfolder, file_path)
                shutil.copy(full_path_in, full_path_out)
