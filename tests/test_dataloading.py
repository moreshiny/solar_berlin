import unittest
import os

import dataloading


class TestSelectMapImages(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = "data"
        self.input_path = os.path.join(self.data_path, "map")
        self.output_path = os.path.join(self.data_path, "split")

        self.train_size = 1024
        self.test_size = 256

    def test_select_map_images_creates_selection_files(self):
        dataloading.select_map_images(
            train_size=1024,
            test_size=256,
            input_path=self.input_path,
            output_path=self.output_path,
        )
        self.assertTrue(os.path.isfile(
            os.path.join(self.output_path, "train_map_selection.csv"),
        ))
        self.assertTrue(os.path.isfile(
            os.path.join(self.output_path, "test_map_selection.csv"),
        ))

    def test_select_map_images_creates_correct_columns_train(self):
        dataloading.select_map_images(
            train_size=self.train_size,
            test_size=self.test_size,
            input_path=self.input_path,
            output_path=self.output_path,
        )
        # load the created csvs
        train_file_name = os.path.join(
            self.output_path, "train_map_selection.csv")
        with open(train_file_name) as f:
            train_lines = f.readlines()
            for train_line in train_lines:
                train_line_split = train_line.replace("\n", "").split(",")
                self.assertEqual(len(train_line_split), 2)
                for image_file in train_line_split:
                    self.assertTrue(
                        os.path.isfile(
                            os.path.join(self.input_path, image_file)
                        )
                    )

    def test_select_map_images_creates_correct_columns_test(self):
        dataloading.select_map_images(
            train_size=self.train_size,
            test_size=self.test_size,
            input_path=self.input_path,
            output_path=self.output_path,
        )
        # load the created csvs
        test_file_name = os.path.join(
            self.output_path, "test_map_selection.csv")
        with open(test_file_name) as f:
            test_lines = f.readlines()
            for test_line in test_lines:
                # split by test_line by "," irgnoring newlines
                test_line_split = test_line.replace("\n", "").split(",")
                self.assertEqual(len(test_line_split), 2)
                for image_file in test_line_split:
                    self.assertTrue(
                        os.path.isfile(
                            os.path.join(self.input_path, image_file)
                        )
                    )

    def test_select_map_images_selects_correct_number_of_images(self):
        dataloading.select_map_images(
            train_size=self.train_size,
            test_size=self.test_size,
            input_path=self.input_path,
            output_path=self.output_path,
        )
        train_file_name = os.path.join(
            self.output_path, "train_map_selection.csv")
        with open(train_file_name) as f:
            train_lines = f.readlines()
            self.assertEqual(len(train_lines), 1024)

        test_file_name = os.path.join(
            self.output_path, "test_map_selection.csv")
        with open(test_file_name) as f:
            test_lines = f.readlines()
            self.assertEqual(len(test_lines), 256)

    def test_select_map_images_selects_only_tifs_train(self):
        dataloading.select_map_images(
            train_size=self.train_size,
            test_size=self.test_size,
            input_path=self.input_path,
            output_path=self.output_path,
        )

        train_file_name = os.path.join(
            self.output_path, "train_map_selection.csv")
        with open(train_file_name) as f:
            train_lines = f.readlines()
            for line in train_lines:
                line_split = line.replace("\n", "").split(",")
                for filename in line_split:
                    self.assertTrue(filename.endswith(".tif"))

    def test_select_map_images_selects_only_tifs_test(self):
        dataloading.select_map_images(
            train_size=self.train_size,
            test_size=self.test_size,
            input_path=self.input_path,
            output_path=self.output_path,
        )

        test_file_name = os.path.join(
            self.output_path, "test_map_selection.csv")
        with open(test_file_name) as f:
            test_lines = f.readlines()
            for line in test_lines:
                line_split = line.replace("\n", "").split(",")
                for filename in line_split:
                    self.assertTrue(filename.endswith(".tif"))

# TODO test files alight correctly

# TODO test files are randomized


if __name__ == '__main__':
    unittest.main()
