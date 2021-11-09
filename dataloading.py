import os
import random


def select_map_images(train_size, test_size, input_path, output_path):

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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

    # create file train_map_selection.csv
    file_name_train = os.path.join(output_path, "train_map_selection.csv")
    with open(file_name_train, 'w') as f:
        for map_file_name, mask_file_name in files_train:
            f.write(map_file_name + "," + mask_file_name + "\n")

    # create file test_map_selection.csv
    file_name_test = os.path.join(output_path, "test_map_selection.csv")
    with open(file_name_test, 'w') as f:
        for map_file_name, mask_file_name in files_test:
            f.write(map_file_name + "," + mask_file_name + "\n")
