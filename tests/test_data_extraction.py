from selection.selection import DataSelector
import unittest
import os
import shutil

from typing import List, Tuple

import random
random.seed(42)


class TestSelectMapImages(unittest.TestCase):
    """Tests for DataSelector._select_random_map_images."""

    def __init__(self, *args, **kwargs):
        """Initialise some shared test parameters."""
        super().__init__(*args, **kwargs)
        self.data_path = "data/testing"
        self.input_path = os.path.join(self.data_path, "map")
        self.output_path = os.path.join(self.data_path, "split")

        self.train_size = 1024
        self.test_size = 256

    # TODO define saner data structure
    def test_select_map_images_returns_a_list_of_lists_of_tuples(self):
        file_lists = DataSelector._select_random_map_images(
            train_size=1024,
            test_size=256,
            input_path=self.input_path,
        )
        self.assertIsInstance(file_lists, List)
        for file_list in file_lists:
            self.assertIsInstance(file_list, List)
            for file_tuple in file_list:
                self.assertIsInstance(file_tuple, Tuple)

    def test_select_map_images_returns_two_file_lists(self):
        file_lists = DataSelector._select_random_map_images(
            train_size=self.train_size,
            test_size=self.test_size,
            input_path=self.input_path,
        )

        self.assertEqual(len(file_lists), 2)

    def test_select_map_images_returns_correct_number_of_file_tuples_train(self):
        file_lists = DataSelector._select_random_map_images(
            train_size=self.train_size,
            test_size=self.test_size,
            input_path=self.input_path,
        )
        self.assertEqual(len(file_lists[0]), self.train_size)

    def test_select_map_images_returns_correct_number_of_file_tuples_test(self):
        file_lists = DataSelector._select_random_map_images(
            train_size=self.train_size,
            test_size=self.test_size,
            input_path=self.input_path,
        )
        self.assertEqual(len(file_lists[0]), self.train_size)

    def test_select_map_images_returns_two_existing_files_per_tuple(self):
        file_lists = DataSelector._select_random_map_images(
            train_size=self.train_size,
            test_size=self.test_size,
            input_path=self.input_path,
        )
        self.assertEqual(len(file_lists), 2)
        for file_list in file_lists:
            for file_tuple in file_list:
                self.assertEqual(len(file_tuple), 2)
                for image_file in file_tuple:
                    self.assertTrue(
                        os.path.isfile(
                            os.path.join(self.input_path, image_file)
                        )
                    )

    def test_select_map_images_selects_only_tifs_train(self):
        file_lists = DataSelector._select_random_map_images(
            train_size=self.train_size,
            test_size=self.test_size,
            input_path=self.input_path,
        )

        for file_list in file_lists:
            for file_tuple in file_list:
                for image_file in file_tuple:
                    self.assertTrue(image_file.endswith(".tif"))

# TODO ensure OutputPathExistsError is raised at appropriate times

# TODO ensure files align correctly (map and mask)

# TODO ensure files are randomized

# TODO ensure files are all of the same dimensions


class TestCopyImageFiles(unittest.TestCase):
    """Tests for DataSelector._copy_image_files."""

    def __init__(self, *args, **kwargs):
        """Initialise some shared test parameters."""
        super().__init__(*args, **kwargs)
        self.data_path = "data/testing"
        self.input_path = os.path.join(self.data_path, "map")
        self.output_path = os.path.join(self.data_path, "test_selected")

    def test_copy_file_copies_file(self):
        """Test that the expected number of files is created in the expected
        location.
        """
        train_size = 10
        test_size = 5
        image_files = DataSelector._select_random_map_images(
            train_size=train_size,
            test_size=test_size,
            input_path=self.input_path
        )

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

        DataSelector._copy_image_files(
            image_files,
            input_path=self.input_path,
            output_path=self.output_path,
        )

        train_path = os.path.join(self.output_path, "train")
        test_path = os.path.join(self.output_path, "test")

        self.assertTrue(os.path.exists(self.output_path))
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(test_path))
        self.assertEqual(len(os.listdir(train_path)), 2*train_size)
        self.assertEqual(len(os.listdir(test_path)), 2*test_size)


if __name__ == '__main__':
    unittest.main()
