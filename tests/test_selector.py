import filecmp
import unittest
import os
import shutil
import glob
import numpy as np
from PIL import Image


from selection.selection import DataExtractor, DataSelector
from common.errors import InvalidPathError, AbsolutePathError, OutputPathExistsError
from common.errors import InvalidTileSizeError, InsuffientDataError


TILE_SIZES = (250, 500, 512)
SELECTION_SIZES = [(10, 5)]
RANDOM_SEED = 42


class TestDataExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_path = os.path.join("data", "testing", "converted_test")
        cls.input_path_invalid_vector = os.path.join(
            "data", "testing", "converted_invalid_test")
        cls.output_path = os.path.join("data", "testing", "extracted")
        cls.output_path_invalid_vector = os.path.join(
            "data", "testing", "extracted_invalid")
        cls.tile_sizes = TILE_SIZES
        cls.clean_up()
        cls.extractors = []
        for tile_size in cls.tile_sizes:
            cls.extractors.append(
                DataExtractor(
                    input_path=cls.input_path,
                    output_path=cls.output_path,
                    tile_size=tile_size,
                    testing=True,  # limit input to 16 tiles for faster testing
                    lossy=True,
                )
            )

    @classmethod
    def clean_up(cls):
        for tile_size in cls.tile_sizes:
            tile_subdir = cls._tile_subdir(tile_size)
            if os.path.exists(os.path.join(cls.output_path, tile_subdir)):
                shutil.rmtree(os.path.join(cls.output_path, tile_subdir))
            if os.path.exists(os.path.join(cls.output_path_invalid_vector, tile_subdir)):
                shutil.rmtree(os.path.join(
                    cls.output_path_invalid_vector, tile_subdir))

    @staticmethod
    def _tile_subdir(tile_size):
        return f"tiles_{tile_size}"

    def test_data_extractor_creates_output_paths(self):
        for tile_size in self.tile_sizes:
            tile_subdir = self._tile_subdir(tile_size)
            self.assertTrue(
                os.path.exists(os.path.join(self.output_path, tile_subdir))
            )

    def test_data_extractor_refuses_to_overwrite_existing_directory(self):
        existing_path = os.path.join(self.output_path, "existing_path")
        with self.assertRaises(OutputPathExistsError):
            DataExtractor(
                input_path=self.input_path,
                output_path=existing_path,
                tile_size=250,  # corresponding directory exists
                testing=True,  # limit input to 16 tiles for faster testing
            )

    def test_data_extractor_verifies_existing_input_directory(self):
        existing_extractor = DataExtractor(
            input_path=self.input_path,
            output_path=self.output_path,
            tile_size=self.tile_sizes[0],
            testing=True,  # limit input to 16 tiles for faster testing
        )
        total_tiles = existing_extractor.total_tiles
        self.assertEqual(total_tiles, 16)

    def test_data_extractor_produces_expected_filenames(self):
        for tile_size in self.tile_sizes:
            tile_subdir = self._tile_subdir(tile_size)
            tile_fns = glob.glob(
                os.path.join(self.output_path, tile_subdir, "*.png")
            )
            for image_fn in tile_fns:
                pattern_msk = "^.*-dop20[0-9_]*_msk.png$"
                pattern_map = "^.*-dop20[0-9_]*_map.png$"

                if "_map.png" in image_fn:
                    self.assertRegex(image_fn, pattern_map)
                else:
                    self.assertRegex(image_fn, pattern_msk)

    def test_data_extractor_produces_expected_map_msk_split(self):
        for tile_size in self.tile_sizes:
            tile_subdir = self._tile_subdir(tile_size)
            tile_fns = glob.glob(
                os.path.join(self.output_path, tile_subdir, "*.png")
            )
            map_count = 0
            msk_count = 0
            for image_fn in tile_fns:
                if "_map.png" in image_fn:
                    map_count += 1
                elif "_msk.png" in image_fn:
                    msk_count += 1

            self.assertEqual(map_count, msk_count)

    def test_data_extractor_produces_expected_image_sizes(self):
        for tile_size in self.tile_sizes:
            tile_subdir = self._tile_subdir(tile_size)
            tile_fns = glob.glob(
                os.path.join(self.output_path, tile_subdir, "*.png")
            )
            for image_fn in tile_fns:
                image = np.array(Image.open(image_fn))
                if "map" in image_fn:
                    self.assertEqual(image.shape, (tile_size, tile_size, 3))
                else:
                    self.assertEqual(image.shape, (tile_size, tile_size))

    def test_data_extractor_raises_error_on_invalid_image_size_0(self):
        with self.assertRaises(InvalidTileSizeError):
            DataExtractor(
                input_path=self.input_path,
                output_path=self.output_path,
                tile_size=0,
                testing=True,  # limit input to 16 tiles for faster testing
            )

    def test_data_extractor_raises_error_on_invalid_image_size_11k(self):
        with self.assertRaises(InvalidTileSizeError):
            DataExtractor(
                input_path=self.input_path,
                output_path=self.output_path,
                tile_size=11000,
                testing=True,  # limit input to 16 tiles for faster testing
            )

    def test_data_extractor_raises_error_on_invalid_image_size_557(self):
        with self.assertRaises(InvalidTileSizeError):
            DataExtractor(
                input_path=self.input_path,
                output_path=self.output_path,
                # lossy is False by default, so this should fail:
                tile_size=557,
                testing=True,  # limit input to 16 tiles for faster testing
            )

    def test_data_extractor_raises_error_on_invalid_input_path(self):
        with self.assertRaises(InvalidPathError):
            DataExtractor(
                input_path="invalid_path",
                output_path=self.output_path,
                tile_size=250,
                testing=True,  # limit input to 16 tiles for faster testing
            )

    def test_data_extractor_raises_error_on_absolute_path_in(self):
        with self.assertRaises(AbsolutePathError):
            DataExtractor(
                input_path=os.path.abspath(self.input_path),
                output_path=self.output_path,
                tile_size=250,
                testing=True,  # limit input to 16 tiles for faster testing
            )

    def test_data_extractor_raises_error_on_absolute_path_out(self):
        with self.assertRaises(AbsolutePathError):
            DataExtractor(
                input_path=self.input_path,
                output_path=os.path.abspath(self.output_path),
                tile_size=250,
                testing=True,  # limit input to 16 tiles for faster testing
            )

    def test_data_extractor_raises_error_on_empty_input_path(self):
        with self.assertRaises(InvalidPathError):
            DataExtractor(
                input_path="",
                output_path=self.output_path,
                tile_size=250,
                testing=True,  # limit input to 16 tiles for faster testing
            )

    def test_data_extractor_produces_masks_with_expected_categories(self):
        all_msk_files = []
        for tile_size in self.tile_sizes:
            all_msk_files += glob.glob(os.path.join(
                self.output_path, self._tile_subdir(tile_size), "*_msk.png"
            ))
        msk_set = set()
        for msk_file in all_msk_files:
            msk_array = np.array(Image.open(msk_file))
            msk_set.update(msk_array.flatten())

        true_set = {0, 63, 127, 191, 255}
        self.assertSetEqual(msk_set, true_set)

    def test_data_extractor_produces_expected_images(self):
        all_files_new = []
        all_files_known = []
        for tile_size in self.tile_sizes:
            all_files_new += glob.glob(
                os.path.join(self.output_path,
                             self._tile_subdir(tile_size), "*.png")
            )
            all_files_known += glob.glob(
                os.path.join(self.output_path + "_test",
                             self._tile_subdir(tile_size), "*.png")
            )
        # check that all files are identical
        print(zip(all_files_new, all_files_known))
        self.assertEqual(len(all_files_new), len(all_files_known))
        for i in range(len(all_files_new)):
            self.assertTrue(filecmp.cmp(all_files_new[i], all_files_known[i]))

    def test_data_extractor_does_not_fail_on_non_existing_vector(self):
        dataextractor = DataExtractor(
            input_path=self.input_path_invalid_vector,
            output_path=self.output_path_invalid_vector,
            tile_size=250,
            testing=True,  # limit input to 16 tiles for faster testing
        )
        self.assertEqual(dataextractor.total_tiles, 16)


class TestDataSelector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # remove output from last run
        cls.tile_sizes = TILE_SIZES

        cls.extractor_input_path = os.path.join(
            "data", "testing", "converted_test")
        cls.input_path = os.path.join("data", "testing", "extracted_test")
        cls.output_path = os.path.join("data", "testing", "selected")
        cls.verfication_path = os.path.join("data", "testing", "selected_test")

        cls.clean_up()


#        cls._first_run = True

        cls.extractors = []
        for tile_size in cls.tile_sizes:
            cls.extractors.append(
                DataExtractor(
                    input_path=cls.extractor_input_path,
                    output_path=cls.input_path,
                    tile_size=tile_size,
                    testing=True,  # limit input to 16 tiles for faster testing
                    lossy=True,
                )
            )

        cls.selected_paths = []

        selection_sizes = SELECTION_SIZES
        for selection_size in selection_sizes:
            for extractor in cls.extractors:
                cls.selector = DataSelector(
                    extractor=extractor,
                    output_path=cls.output_path,
                    train_n=selection_size[0],
                    test_n=selection_size[1],
                    random_seed=RANDOM_SEED,
                )
                cls.selected_paths.append(os.path.join(
                    cls.output_path,
                    cls.selected_subdir(
                        tile_size,
                        selection_size[0],
                        selection_size[1],
                        RANDOM_SEED,
                    ),
                ))

    @classmethod
    def selected_subdir(cls, tile_size, train_n, test_n, random_seed):
        return f"selected_tiles"\
            + f"_{tile_size}"\
            + f"_{train_n}"\
            + f"_{test_n}"\
            + f"_{random_seed}"\


    @classmethod
    def clean_up(cls):
        for tile_size in cls.tile_sizes:
            for selection_size in SELECTION_SIZES:
                subdir = cls.selected_subdir(
                    tile_size,
                    selection_size[0],
                    selection_size[1],
                    RANDOM_SEED,
                )
                if os.path.exists(os.path.join(cls.output_path, subdir)):
                    shutil.rmtree(os.path.join(cls.output_path, subdir))

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
                train_files = os.listdir(os.path.join(selected_path, "train"))
                train_files_no = len(train_files) // 2
                test_files = os.listdir(os.path.join(selected_path, "test"))
                test_files_no = len(test_files) // 2
                self.assertEqual(train_files_no, selection_size[0])
                self.assertEqual(test_files_no, selection_size[1])

    def test_data_selector_refuses_to_overwrite_existing_directory(self):
        existing_path = os.path.join(
            self.output_path,
            "existing_path"
        )
        with self.assertRaises(OutputPathExistsError):
            DataSelector(
                extractor=self.extractors[0],
                output_path=existing_path,
                train_n=10,
                test_n=5,
                random_seed=RANDOM_SEED,
            )

    def test_data_selector_throws_error_more_images_requested_than_exist(self):
        # huge train_n (only 16 tiles are available during testing)
        train_n = 10000
        test_n = 5
        with self.assertRaises(InsuffientDataError):
            DataSelector(
                extractor=self.extractors[0],
                output_path=self.output_path,
                train_n=train_n,
                test_n=test_n,
                random_seed=RANDOM_SEED,
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

    def test_data_selector_raises_error_on_empty_output_path(self):
        with self.assertRaises(InvalidPathError):
            DataSelector(
                extractor=self.extractors[0],
                output_path="",
                train_n=10,
                test_n=5,
                random_seed=RANDOM_SEED,
            )

    def test_data_selector_raises_error_on_absolute_output_path(self):
        with self.assertRaises(AbsolutePathError):
            DataSelector(
                extractor=self.extractors[0],
                output_path=os.path.abspath(self.output_path),
                train_n=10,
                test_n=5,
                random_seed=RANDOM_SEED,
            )

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

    def test_data_selector_produces_expected_images(self):
        for selected_path in self.selected_paths:
            all_files_new = glob.glob(
                os.path.join(selected_path, "**", "*.png"),
                recursive=True,
            )
            true_files_path = os.path.join(
                self.verfication_path, os.path.basename(selected_path)
            )
            all_files_known = glob.glob(
                os.path.join(true_files_path, "**", "*.png"),
                recursive=True,
            )
        # check that all files are identical
        self.assertEqual(len(all_files_new), len(all_files_known))
        for i in range(len(all_files_new)):
            self.assertTrue(filecmp.cmp(all_files_new[i], all_files_known[i]))

    def test_data_selector_picks_different_images_for_different_random_seeds(self):
        random_seeds = (43, 44)
        tile_size = 250
        train_n = 10
        test_n = 5

        selected_images = []

        for random_seed in random_seeds:
            selected_path = os.path.join(
                self.output_path,
                f"selected_tiles_{tile_size}_{train_n}_{test_n}_{random_seed}"
            )
            if os.path.exists(selected_path):
                shutil.rmtree(selected_path)
            DataSelector(
                # first extractor has tile size 250
                extractor=self.extractors[0],
                output_path=selected_path,
                train_n=train_n,
                test_n=test_n,
                random_seed=random_seed,
            )

            selected_images.append(glob.glob(
                os.path.join(selected_path, "**", "*.png"),
                recursive=True,
            ))

        duplicates = set(selected_images[0]).intersection(selected_images[1])
        self.assertEqual(len(duplicates), 0)

    def test_data_selector_does_not_pick_same_images_in_test_and_train(self):
        for selected_path in self.selected_paths:
            train_fns = glob.glob(
                os.path.join(selected_path, "train", "**", "*.png"),
            )
            test_fns = glob.glob(
                os.path.join(selected_path, "test", "**", "*.png"),
            )

            for test_file in test_fns:
                self.assertNotIn(test_file, train_fns)


if __name__ == "__main__":
    unittest.main()
