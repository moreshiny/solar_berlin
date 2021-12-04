# basic unittest structure
import unittest
import os
import shutil
import glob
import numpy as np
from PIL import Image


from extraction.selection import DataSelector

INPUT_PATH = os.path.join("data", "testing", "converted")
OUTPUT_PATH = os.path.join("data", "testing", "selected")

TILE_SIZES = (250, 500)
SELECTION_SIZES = ((10, 5), )
RANDOM_SEED = 42


class TestSelection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.clean_up()

        cls._first_run = True
        cls.selector = DataSelector(
            input_path=INPUT_PATH,
            testing=True,  # limit input to 32 tiles for faster testing
        )

        cls.selected_paths = []

        for selection_size in SELECTION_SIZES:
            for tile_size in TILE_SIZES:
                cls.selector.select_data(
                    tile_size=tile_size,
                    output_path=OUTPUT_PATH,
                    train_n=selection_size[0],
                    test_n=selection_size[1],
                    random_seed=RANDOM_SEED,
                )
                cls.selected_paths.append(os.path.join(
                    OUTPUT_PATH,
                    f"selected_tiles"
                    + f"_{tile_size}"
                    + f"_{selection_size[0]}"
                    + f"_{selection_size[1]}"
                    + f"_{RANDOM_SEED}"
                ))

    @staticmethod
    def clean_up():
        for tile_size in TILE_SIZES:
            tile_path = os.path.join(INPUT_PATH, f"tiled_{tile_size}")
            if os.path.exists(tile_path):
                shutil.rmtree(tile_path)
            for selection_size in SELECTION_SIZES:
                selection_path = os.path.join(
                    OUTPUT_PATH,
                    f"selected_tiles_"
                    + f"{tile_size}"
                    + f"_{selection_size[0]}"
                    + f"_{selection_size[1]}"
                    + f"_{RANDOM_SEED}"
                )
                if os.path.exists(selection_path):
                    shutil.rmtree(selection_path)

    def test_data_selector_creates_output_paths(self):
        for selected_path in self.selected_paths:
            self.assertTrue(os.path.exists(
                os.path.join(selected_path, "train")))
            self.assertTrue(os.path.exists(
                os.path.join(selected_path, "test")))

    def test_data_selector_selects_requested_number_of_images(self):
        for selected_path in self.selected_paths:
            for selection_size in SELECTION_SIZES:
                # count of train/test files is half the total due to map/msk
                train_files_no = len(os.listdir(
                    os.path.join(selected_path, "train"))) // 2
                test_files_no = len(os.listdir(
                    os.path.join(selected_path, "test"))) // 2
                self.assertEqual(train_files_no, selection_size[0])
                self.assertEqual(test_files_no, selection_size[1])

    def test_data_selector_throws_error_more_images_requested_than_exist(self):
        # huge train_n (only 32 tiles are available during testing)
        train_n = 10000
        test_n = 5
        tile_size = 500

        with self.assertRaises(ValueError):

            self.selector.select_data(
                tile_size=tile_size,
                train_n=train_n,
                test_n=test_n,
                output_path=OUTPUT_PATH,
                random_seed=42,
                multiclass=True,
            )

    def test_data_selector_produces_expected_filenames(self):
        for selected_path in self.selected_paths:
            train_fns = os.listdir(os.path.join(selected_path, "train"))
            test_fns = os.listdir(os.path.join(selected_path, "test"))

            for image_fn in train_fns + test_fns:
                pattern_msk = "^.*-dop20[0-9_]*_msk.png$"
                pattern_map = "^.*-dop20[0-9_]*_map.png$"

                if "_map.png" in image_fn:
                    self.assertRegex(image_fn, pattern_map)
                else:
                    self.assertRegex(image_fn, pattern_msk)

    def test_data_selector_produces_expected_map_msk_split(self):
        for selected_path in self.selected_paths:
            train_files = os.listdir(os.path.join(selected_path, "train"))
            test_files = os.listdir(os.path.join(selected_path, "test"))

            map_count = 0
            msk_count = 0
            for image_fn in train_files + test_files:
                if "_map.png" in image_fn:
                    map_count += 1
                elif "_msk.png" in image_fn:
                    msk_count += 1

            self.assertEqual(map_count, msk_count)

    def test_data_selector_produces_expected_image_sizes(self):
        for selected_path in self.selected_paths:
            all_files = glob.glob(
                os.path.join(selected_path, "**", "*.png"),
                recursive=True,
            )

            tile_size = int(selected_path.split("_")[2])

            for image_fn in all_files:
                image = np.array(Image.open(image_fn))
                if "map" in image_fn:
                    self.assertEqual(image.shape, (tile_size, tile_size, 3))
                else:
                    self.assertEqual(image.shape, (tile_size, tile_size))

    def test_data_selector_raises_error_on_invalid_image_size_0(self):
        with self.assertRaises(ValueError):
            self.selector.select_data(0, 10, 5, OUTPUT_PATH, 42)

    def test_data_selector_raises_error_on_invalid_image_size_11k(self):
        with self.assertRaises(ValueError):
            self.selector.select_data(11_000, 10, 5, OUTPUT_PATH, 42)

    def test_data_selector_raises_error_on_invalid_image_size_224(self):
        with self.assertRaises(ValueError):
            self.selector.select_data(224, 10, 5, OUTPUT_PATH, 42)

    def test_data_selector_raises_error_on_invalid_input_path(self):
        with self.assertRaises(FileNotFoundError):
            DataSelector(
                input_path="invalid_path",
            )

    def test_data_selector_raises_error_on_absolute_path(self):
        with self.assertRaises(ValueError):
            DataSelector(
                input_path=os.path.abspath(INPUT_PATH),
            )

    def test_data_selector_raises_error_on_empty_input_path(self):
        with self.assertRaises(FileNotFoundError):
            DataSelector(
                input_path="",
            )

    def test_data_selector_raises_error_on_absolute_output_path(self):
        with self.assertRaises(ValueError):
            self.selector.select_data(
                500, 10, 5, output_path=os.path.abspath(OUTPUT_PATH),)

    def test_data_selector_produces_masks_with_expected_categories(self):
        all_msk_files = []
        for selected_path in self.selected_paths:
            all_msk_files += glob.glob(os.path.join(
                selected_path, "**", "*_msk.png"), recursive=True)
        msk_set = set()
        for msk_file in all_msk_files:
            msk_array = np.array(Image.open(msk_file))
            msk_set.update(msk_array.flatten())

        true_set = {0, 63, 127, 191, 255}
        self.assertSetEqual(msk_set, true_set)


if __name__ == "__main__":
    unittest.main()
