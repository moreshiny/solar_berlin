import os
import random
import shutil


def select_map_images(train_size, test_size, input_path):

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

    files_train = files_zipped[:train_size]
    files_test = files_zipped[train_size:train_size+test_size]

    return [files_train, files_test]


def copy_image_files(image_files, input_path, output_path, delete_existing=False):

    if delete_existing and os.path.exists(output_path):
        shutil.rmtree(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise Exception("At least one of the output directory already exists."
                        "\nSet delete_existing=True to remove it.")

    files = {}
    files["train"] = image_files[0]
    files["test"] = image_files[1]

    for subfolder in ["train", "test"]:
        output_path_subfolder = os.path.join(output_path, subfolder)
        if not os.path.exists(output_path_subfolder):
            os.makedirs(output_path_subfolder)
        for file_tuple in files[subfolder]:
            for file_path in file_tuple:
                full_path_in = os.path.join(input_path, file_path)
                full_path_out = os.path.join(output_path_subfolder, file_path)
                shutil.copy(full_path_in, full_path_out)
