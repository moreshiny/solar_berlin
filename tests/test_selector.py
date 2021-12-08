<<<<<<< HEAD
import filecmp
=======
<<<<<<< HEAD
<<<<<<< HEAD
import filecmp
=======
# basic unittest structure
<<<<<<< HEAD
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======
=======
>>>>>>> b0c8908 (Use custom exception names)
import filecmp
>>>>>>> 1658920 (Fix binary mask loading and add mask category test cases)
>>>>>>> 5acb817 (Refactor data selector tests)
import unittest
import os
import shutil
import glob
import numpy as np
from PIL import Image


<<<<<<< HEAD
<<<<<<< HEAD
from roof.selection import DataExtractor, DataSelector
from roof.errors import (
    AbsolutePathError,
    InsuffientDataError,
    InvalidPathError,
    InvalidTileSizeError,
    OutputPathExistsError,
)


TILE_SIZES = (250, 500, 512)
SELECTION_SIZES = [(10, 5)]
RANDOM_SEED = 42


class TestDataExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_path = os.path.join(
            "data", "testing", "converted_test"
        )
        cls.input_path_invalid_vector = os.path.join(
            "data", "testing", "converted_invalid_test"
        )
        cls.input_path_incomplete_extraction = os.path.join(
            "data", "testing", "converted_coco_missing_test"
        )
        cls.output_path = os.path.join(
            "data", "testing", "extracted"
        )
        cls.output_path_invalid_vector = os.path.join(
            "data", "testing", "extracted_invalid"
        )
        cls.output_path_incomplete_extraction = os.path.join(
            "data", "testing", "extracted_coco_missing_test"
        )
        cls.tile_sizes = TILE_SIZES

        # clean up here so we can leave results in place to inspect at the end
        cls._clean_up()

        # create one extracter for each tile size for use by tests
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
    def _clean_up(cls):
        for tile_size in cls.tile_sizes:
            tile_subdir = cls._tile_subdir(tile_size)
            for base_path in [cls.output_path, cls.output_path_invalid_vector]:
                tile_path = os.path.join(base_path, tile_subdir)
                if os.path.exists(tile_path):
                    shutil.rmtree(tile_path)

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
                self.output_path, self._tile_subdir(tile_size), "**", "*_msk.png"
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
                             self._tile_subdir(tile_size), "**", "*.png")
            )
            all_files_known += glob.glob(
                os.path.join(self.output_path + "_test",
                             self._tile_subdir(tile_size), "**", "*.png")
            )
        all_files_new.sort()
        all_files_known.sort()

        # check that all files are identical
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

    def test_data_extractor_saves_produced_tiles_in_subfolders(self):
        for tile_size in self.tile_sizes:
            tile_subdir = self._tile_subdir(tile_size)
            raster_names = glob.glob(self.input_path + "/*.tif")
            for raster_name in raster_names:
                raster_basename = os.path.basename(raster_name)
                raster_basename_wo_ext = os.path.splitext(raster_basename)[0]
                self.assertTrue(os.path.exists(os.path.join(
                    self.output_path, tile_subdir, raster_basename_wo_ext)))

    def test_data_extractor_creates_expected_coco_json(self):
        coco_new_fns = []
        coco_known_fns = []
        for tile_size in self.tile_sizes:
            coco_new_fns += glob.glob(
                os.path.join(self.output_path,
                             self._tile_subdir(tile_size), "**", "*.json")
            )
            coco_known_fns += glob.glob(
                os.path.join(self.output_path + "_test",
                             self._tile_subdir(tile_size), "**", "*.json")
            )
        coco_new_fns.sort()
        coco_known_fns.sort()

        # check that all files are identical
        self.assertEqual(len(coco_new_fns), len(coco_known_fns))
        for i in range(len(coco_new_fns)):
            self.assertTrue(filecmp.cmp(coco_new_fns[i], coco_known_fns[i]))

    def test_data_extractor_creates_missing_coco_json_existing_tiles(self):

        coco_to_clean = glob.glob(os.path.join(
            self.output_path_incomplete_extraction, "**", "*.json"), recursive=True)

        for coco_fn in coco_to_clean:
            os.remove(coco_fn)

        temp_folders = glob.glob(os.path.join(
            self.output_path_incomplete_extraction, "**", "temp"), recursive=True)

        for temp_folder in temp_folders:
            shutil.rmtree(temp_folder)

        _ = DataExtractor(
            input_path=self.input_path,
            output_path=self.output_path_incomplete_extraction,
            tile_size=250,
            testing=True,  # limit input to 16 tiles for faster testing
            lossy=True,
        )
        coco_new_fns = glob.glob(
            os.path.join(self.output_path_incomplete_extraction,
                         self._tile_subdir(250), "**", "*.json")
        )

        coco_known_fns = glob.glob(
            os.path.join(self.output_path + "_test",
                         self._tile_subdir(250), "**", "*.json")
        )
        coco_new_fns.sort()
        coco_known_fns.sort()
        # check that all files are identical
        self.assertEqual(len(coco_new_fns), len(coco_known_fns))
        for i in range(len(coco_new_fns)):
            self.assertTrue(filecmp.cmp(coco_new_fns[i], coco_known_fns[i]))

class TestDataSelector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # remove output from last run
        cls.tile_sizes = TILE_SIZES

        cls.extractor_input_path = os.path.join(
            "data", "testing", "converted_test"
        )
        cls.input_path = os.path.join(
            "data", "testing", "extracted_test"
        )
        cls.output_path = os.path.join(
            "data", "testing", "selected"
        )
        cls.verfication_path = os.path.join(
            "data", "testing", "selected_test"
        )

        # clean up here so we can leave output in place at the end
        cls._clean_up()

        # test tiles of all sizes exists, so these extractors simply verify that
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

        # determine and store the paths to be created separately for testing
        cls.selected_paths = []
        selection_sizes = SELECTION_SIZES
        for selection_size in selection_sizes:
            for extractor in cls.extractors:
                cls.selector = DataSelector(
                    extractor=extractor,
                    output_path=cls.output_path,
=======
<<<<<<< HEAD
from extraction.selection import DataSelector
<<<<<<< HEAD
<<<<<<< HEAD
from extraction.selection import InvalidPathError, AbsolutePathError
from extraction.selection import InvalidTileSizeError, InsuffientDataError
=======
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======
from extraction.selection import InvalidPathError, AbsolutePathError
from extraction.selection import InvalidTileSizeError, InsuffientDataError
>>>>>>> b0c8908 (Use custom exception names)
=======
from selection.selection import DataSelector
=======
from selection.selection import DataExtractor, DataSelector
<<<<<<< HEAD
>>>>>>> 54fa04d (Refactor: separate extraction and selection)
from selection.errors import InvalidPathError, AbsolutePathError, OutputPathExistsError
from selection.errors import InvalidTileSizeError, InsuffientDataError
>>>>>>> 3db5d22 (Separate selection from extraction)
=======
from common.errors import InvalidPathError, AbsolutePathError, OutputPathExistsError
from common.errors import InvalidTileSizeError, InsuffientDataError
>>>>>>> b8e175c (Rename subfolders)


<<<<<<< HEAD
TILE_SIZES = (250, 500)
<<<<<<< HEAD
SELECTION_SIZES = ((10, 5),)
=======
SELECTION_SIZES = ((10, 5), )
>>>>>>> 4241abc (First working version of data selector with multiclass)
RANDOM_SEED = 42


<<<<<<<< HEAD:tests/test_selector.py
=======
TILE_SIZES = (250, 500, 512)
SELECTION_SIZES = [(10, 5)]
RANDOM_SEED = 42


class TestDataExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_path = os.path.join(
            "data", "testing", "converted_test"
        )
        cls.input_path_invalid_vector = os.path.join(
            "data", "testing", "converted_invalid_test"
        )
        cls.output_path = os.path.join(
            "data", "testing", "extracted"
        )
        cls.output_path_invalid_vector = os.path.join(
            "data", "testing", "extracted_invalid"
        )
        cls.tile_sizes = TILE_SIZES

        # clean up here so we can leave results in place to inspect at the end
        cls._clean_up()

        # create one extracter for each tile size for use by tests
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
    def _clean_up(cls):
        for tile_size in cls.tile_sizes:
            tile_subdir = cls._tile_subdir(tile_size)
            for base_path in [cls.output_path, cls.output_path_invalid_vector]:
                tile_path = os.path.join(base_path, tile_subdir)
                if os.path.exists(tile_path):
                    shutil.rmtree(tile_path)

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
        all_files_new.sort()
        all_files_known.sort()

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


>>>>>>> 54fa04d (Refactor: separate extraction and selection)
class TestDataSelector(unittest.TestCase):
========
class TestSelection(unittest.TestCase):
<<<<<<< HEAD
    @classmethod
    def setUpClass(cls):

        # remove output from last run
=======
>>>>>>>> 5acb817 (Refactor data selector tests):tests/test_selection.py

    @classmethod
    def setUpClass(cls):
<<<<<<< HEAD

<<<<<<< HEAD
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======
        # remove output from last run
>>>>>>> b0c8908 (Use custom exception names)
=======
        # remove output from last run
        cls.tile_sizes = TILE_SIZES

        cls.extractor_input_path = os.path.join(
<<<<<<< HEAD
            "data", "testing", "converted_test")
        cls.input_path = os.path.join("data", "testing", "extracted_test")
        cls.output_path = os.path.join("data", "testing", "selected")
        cls.verfication_path = os.path.join("data", "testing", "selected_test")

>>>>>>> 54fa04d (Refactor: separate extraction and selection)
        cls.clean_up()

=======
            "data", "testing", "converted_test"
        )
        cls.input_path = os.path.join(
            "data", "testing", "extracted_test"
        )
        cls.output_path = os.path.join(
            "data", "testing", "selected"
        )
        cls.verfication_path = os.path.join(
            "data", "testing", "selected_test"
        )
>>>>>>> 70307a8 (Clean up tests)

        # clean up here so we can leave output in place at the end
        cls._clean_up()

        # test tiles of all sizes exists, so these extractors simply verify that
        cls.extractors = []
        for tile_size in cls.tile_sizes:
            cls.extractors.append(
                DataExtractor(
                    input_path=cls.extractor_input_path,
                    output_path=cls.input_path,
                    tile_size=tile_size,
<<<<<<< HEAD
                    output_path=OUTPUT_PATH,
>>>>>>> 5acb817 (Refactor data selector tests)
=======
                    testing=True,  # limit input to 16 tiles for faster testing
                    lossy=True,
                )
            )

        # determine and store the paths to be created separately for testing
        cls.selected_paths = []
        selection_sizes = SELECTION_SIZES
        for selection_size in selection_sizes:
            for extractor in cls.extractors:
                cls.selector = DataSelector(
                    extractor=extractor,
                    output_path=cls.output_path,
>>>>>>> 54fa04d (Refactor: separate extraction and selection)
                    train_n=selection_size[0],
                    test_n=selection_size[1],
                    random_seed=RANDOM_SEED,
                )
<<<<<<< HEAD
                cls.selected_paths.append(os.path.join(
                    cls.output_path,
                    cls._selected_subdir(
                        tile_size,
                        selection_size[0],
                        selection_size[1],
                        RANDOM_SEED,
                    ),
                ))

    @classmethod
    def _selected_subdir(cls, tile_size, train_n, test_n, random_seed):
        return f"selected_tiles"\
            + f"_{tile_size}"\
            + f"_{train_n}"\
            + f"_{test_n}"\
            + f"_{random_seed}"\


    @classmethod
    def _clean_up(cls):
        for tile_size in cls.tile_sizes:
            for selection_size in SELECTION_SIZES:
                subdir = cls._selected_subdir(
                    tile_size,
                    selection_size[0],
                    selection_size[1],
                    RANDOM_SEED,
                )
                if os.path.exists(os.path.join(cls.output_path, subdir)):
                    shutil.rmtree(os.path.join(cls.output_path, subdir))

    def test_data_selector_creates_output_paths(self):
        for selected_path in self.selected_paths:
=======
<<<<<<< HEAD
                cls.selected_paths.append(
                    os.path.join(
                        OUTPUT_PATH,
                        f"selected_tiles"
                        + f"_{tile_size}"
                        + f"_{selection_size[0]}"
                        + f"_{selection_size[1]}"
                        + f"_{RANDOM_SEED}",
                    )
                )
=======
                cls.selected_paths.append(os.path.join(
                    cls.output_path,
                    cls._selected_subdir(
                        tile_size,
                        selection_size[0],
                        selection_size[1],
                        RANDOM_SEED,
                    ),
                ))
>>>>>>> 4241abc (First working version of data selector with multiclass)

    @classmethod
    def _selected_subdir(cls, tile_size, train_n, test_n, random_seed):
        return f"selected_tiles"\
            + f"_{tile_size}"\
            + f"_{train_n}"\
            + f"_{test_n}"\
            + f"_{random_seed}"\


    @classmethod
    def _clean_up(cls):
        for tile_size in cls.tile_sizes:
            for selection_size in SELECTION_SIZES:
<<<<<<< HEAD
<<<<<<< HEAD
                selection_path = os.path.join(
                    OUTPUT_PATH,
                    f"selected_tiles_"
                    + f"{tile_size}"
                    + f"_{selection_size[0]}"
                    + f"_{selection_size[1]}"
<<<<<<< HEAD
                    + f"_{RANDOM_SEED}",
=======
                    + f"_{RANDOM_SEED}"
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======
                subdir = cls.selected_subdir(
=======
                subdir = cls._selected_subdir(
>>>>>>> 70307a8 (Clean up tests)
                    tile_size,
                    selection_size[0],
                    selection_size[1],
                    RANDOM_SEED,
>>>>>>> 54fa04d (Refactor: separate extraction and selection)
                )
                if os.path.exists(os.path.join(cls.output_path, subdir)):
                    shutil.rmtree(os.path.join(cls.output_path, subdir))

    def test_data_selector_creates_output_paths(self):
        for selected_path in self.selected_paths:
<<<<<<< HEAD
            self.assertTrue(os.path.exists(os.path.join(selected_path, "train")))
            self.assertTrue(os.path.exists(os.path.join(selected_path, "test")))
=======
>>>>>>> 5acb817 (Refactor data selector tests)
            self.assertTrue(os.path.exists(
                os.path.join(selected_path, "train")))
            self.assertTrue(os.path.exists(
                os.path.join(selected_path, "test")))
<<<<<<< HEAD
=======
>>>>>>> 4241abc (First working version of data selector with multiclass)
>>>>>>> 5acb817 (Refactor data selector tests)

    def test_data_selector_selects_requested_number_of_images(self):
        for selected_path in self.selected_paths:
            for selection_size in SELECTION_SIZES:
                # count of train/test files is half the total due to map/msk
<<<<<<< HEAD
=======
<<<<<<<< HEAD:tests/test_selector.py
>>>>>>> 5acb817 (Refactor data selector tests)
                train_files = os.listdir(os.path.join(selected_path, "train"))
                train_files_no = len(train_files) // 2
                test_files = os.listdir(os.path.join(selected_path, "test"))
                test_files_no = len(test_files) // 2
<<<<<<< HEAD
=======
========
<<<<<<< HEAD
                train_files_no = (
                    len(os.listdir(os.path.join(selected_path, "train"))) // 2
                )
                test_files_no = (
                    len(os.listdir(os.path.join(selected_path, "test"))) // 2
                )
=======
                train_files_no = len(os.listdir(
                    os.path.join(selected_path, "train"))) // 2
                test_files_no = len(os.listdir(
                    os.path.join(selected_path, "test"))) // 2
>>>>>>> 4241abc (First working version of data selector with multiclass)
>>>>>>>> 5acb817 (Refactor data selector tests):tests/test_selection.py
>>>>>>> 5acb817 (Refactor data selector tests)
                self.assertEqual(train_files_no, selection_size[0])
                self.assertEqual(test_files_no, selection_size[1])

    def test_data_selector_refuses_to_overwrite_existing_directory(self):
        existing_path = os.path.join(
<<<<<<< HEAD
<<<<<<< HEAD
            self.output_path,
=======
            OUTPUT_PATH,
>>>>>>> 5acb817 (Refactor data selector tests)
=======
            self.output_path,
>>>>>>> 54fa04d (Refactor: separate extraction and selection)
            "existing_path"
        )
        with self.assertRaises(OutputPathExistsError):
            DataSelector(
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
                input_path=INPUT_PATH,
            ).select_data(
                tile_size=250,
=======
                extractor=self.extractors[0],
>>>>>>> 54fa04d (Refactor: separate extraction and selection)
                output_path=existing_path,
                train_n=10,
                test_n=5,
                random_seed=RANDOM_SEED,
            )

    def test_data_selector_throws_error_more_images_requested_than_exist(self):
        # huge train_n (only 16 tiles are available during testing)
        train_n = 10000
        test_n = 5
<<<<<<< HEAD
        tile_size = 500

<<<<<<< HEAD
<<<<<<< HEAD
        with self.assertRaises(InsuffientDataError):
=======
        with self.assertRaises(ValueError):
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======
        with self.assertRaises(InsuffientDataError):
>>>>>>> b0c8908 (Use custom exception names)

            self.selector.select_data(
                tile_size=tile_size,
                train_n=train_n,
                test_n=test_n,
                output_path=OUTPUT_PATH,
                random_seed=42,
>>>>>>> 5acb817 (Refactor data selector tests)
=======
        with self.assertRaises(InsuffientDataError):
            DataSelector(
                extractor=self.extractors[0],
                output_path=self.output_path,
                train_n=train_n,
                test_n=test_n,
                random_seed=RANDOM_SEED,
>>>>>>> 54fa04d (Refactor: separate extraction and selection)
            )

    def test_data_selector_produces_expected_filenames(self):
        for selected_path in self.selected_paths:
            train_fns = os.listdir(os.path.join(selected_path, "train"))
            test_fns = os.listdir(os.path.join(selected_path, "test"))

            for image_fn in train_fns + test_fns:
                pattern_msk = "^.*-dop20[0-9_]*_msk.png$"
                pattern_map = "^.*-dop20[0-9_]*_map.png$"
<<<<<<< HEAD
                if ".json" in image_fn:
                    continue
=======

>>>>>>> 5acb817 (Refactor data selector tests)
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

<<<<<<< HEAD
<<<<<<< HEAD
            # determine the tile_size from the folder name
=======
>>>>>>> 5acb817 (Refactor data selector tests)
=======
            # determine the tile_size from the folder name
>>>>>>> 70307a8 (Clean up tests)
            tile_size = int(selected_path.split("_")[2])

            for image_fn in all_files:
                image = np.array(Image.open(image_fn))
                if "map" in image_fn:
                    self.assertEqual(image.shape, (tile_size, tile_size, 3))
                else:
                    self.assertEqual(image.shape, (tile_size, tile_size))

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
    def test_data_selector_raises_error_on_invalid_image_size_0(self):
<<<<<<< HEAD
<<<<<<< HEAD
        with self.assertRaises(InvalidTileSizeError):
            self.selector.select_data(0, 10, 5, OUTPUT_PATH, 42)

    def test_data_selector_raises_error_on_invalid_image_size_11k(self):
        with self.assertRaises(InvalidTileSizeError):
=======
        with self.assertRaises(ValueError):
            self.selector.select_data(0, 10, 5, OUTPUT_PATH, 42)

    def test_data_selector_raises_error_on_invalid_image_size_11k(self):
        with self.assertRaises(ValueError):
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======
        with self.assertRaises(InvalidTileSizeError):
            self.selector.select_data(0, 10, 5, OUTPUT_PATH, 42)

    def test_data_selector_raises_error_on_invalid_image_size_11k(self):
        with self.assertRaises(InvalidTileSizeError):
>>>>>>> b0c8908 (Use custom exception names)
            self.selector.select_data(11_000, 10, 5, OUTPUT_PATH, 42)

    def test_data_selector_raises_error_on_invalid_image_size_224(self):
        with self.assertRaises(InvalidTileSizeError):
            # lossy is False by default, so this should fail
            self.selector.select_data(224, 10, 5, OUTPUT_PATH, 42)

    def test_data_selector_raises_error_on_invalid_input_path(self):
<<<<<<< HEAD
<<<<<<< HEAD
        with self.assertRaises(InvalidPathError):
=======
        with self.assertRaises(FileNotFoundError):
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======
=======
    def test_data_selector_raises_error_on_empty_output_path(self):
>>>>>>> 54fa04d (Refactor: separate extraction and selection)
        with self.assertRaises(InvalidPathError):
>>>>>>> b0c8908 (Use custom exception names)
            DataSelector(
                extractor=self.extractors[0],
                output_path="",
                train_n=10,
                test_n=5,
                random_seed=RANDOM_SEED,
            )

<<<<<<< HEAD
    def test_data_selector_raises_error_on_absolute_path(self):
<<<<<<< HEAD
<<<<<<< HEAD
=======
    def test_data_selector_raises_error_on_absolute_output_path(self):
>>>>>>> 54fa04d (Refactor: separate extraction and selection)
        with self.assertRaises(AbsolutePathError):
=======
        with self.assertRaises(ValueError):
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======
        with self.assertRaises(AbsolutePathError):
>>>>>>> b0c8908 (Use custom exception names)
            DataSelector(
<<<<<<< HEAD
                input_path=os.path.abspath(INPUT_PATH),
            )

    def test_data_selector_raises_error_on_empty_input_path(self):
<<<<<<< HEAD
<<<<<<< HEAD
        with self.assertRaises(InvalidPathError):
=======
        with self.assertRaises(FileNotFoundError):
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======
        with self.assertRaises(InvalidPathError):
>>>>>>> b0c8908 (Use custom exception names)
            DataSelector(
                input_path="",
            )

    def test_data_selector_raises_error_on_absolute_output_path(self):
<<<<<<< HEAD
<<<<<<< HEAD
        with self.assertRaises(AbsolutePathError):
            self.selector.select_data(
                500,
                10,
                5,
                output_path=os.path.abspath(OUTPUT_PATH),
            )
=======
        with self.assertRaises(ValueError):
=======
        with self.assertRaises(AbsolutePathError):
>>>>>>> b0c8908 (Use custom exception names)
            self.selector.select_data(
                500, 10, 5, output_path=os.path.abspath(OUTPUT_PATH),)
>>>>>>> 4241abc (First working version of data selector with multiclass)

=======
                extractor=self.extractors[0],
                output_path=os.path.abspath(self.output_path),
                train_n=10,
                test_n=5,
                random_seed=RANDOM_SEED,
            )

>>>>>>> 54fa04d (Refactor: separate extraction and selection)
    def test_data_selector_produces_masks_with_expected_categories(self):
        all_msk_files = []
        for selected_path in self.selected_paths:
<<<<<<< HEAD
            all_msk_files += glob.glob(
                os.path.join(selected_path, "**", "*_msk.png"), recursive=True
            )
=======
            all_msk_files += glob.glob(os.path.join(
                selected_path, "**", "*_msk.png"), recursive=True)
>>>>>>> 4241abc (First working version of data selector with multiclass)
>>>>>>> 5acb817 (Refactor data selector tests)
        msk_set = set()
        for msk_file in all_msk_files:
            msk_array = np.array(Image.open(msk_file))
            msk_set.update(msk_array.flatten())

        true_set = {0, 63, 127, 191, 255}
        self.assertSetEqual(msk_set, true_set)

<<<<<<< HEAD
    def test_data_selector_produces_expected_images(self):
        for selected_path in self.selected_paths:
            all_files_new = sorted(glob.glob(
                os.path.join(selected_path, "**", "*.png"),
                recursive=True,
            ))
            true_files_path = os.path.join(
                self.verfication_path, os.path.basename(selected_path)
            )
            all_files_known = sorted(glob.glob(
                os.path.join(true_files_path, "**", "*.png"),
                recursive=True,
            ))
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 1658920 (Fix binary mask loading and add mask category test cases)
    def test_data_selector_produces_expected_images(self):
        for selected_path in self.selected_paths:
            all_files_new = sorted(glob.glob(
                os.path.join(selected_path, "**", "*.png"),
                recursive=True,
            ))
            true_files_path = os.path.join(
                self.verfication_path, os.path.basename(selected_path)
            )
            all_files_known = sorted(glob.glob(
                os.path.join(true_files_path, "**", "*.png"),
                recursive=True,
<<<<<<< HEAD
            )
>>>>>>> 5acb817 (Refactor data selector tests)
=======
            ))
>>>>>>> 31cb041 (Add sorting to tests to avoid spurious failures)
        # check that all files are identical
        self.assertEqual(len(all_files_new), len(all_files_known))
        for i in range(len(all_files_new)):
            self.assertTrue(filecmp.cmp(all_files_new[i], all_files_known[i]))

<<<<<<< HEAD
<<<<<<< HEAD
    def test_data_selector_picks_different_images_for_different_random_seeds(self):
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======
>>>>>>> 1658920 (Fix binary mask loading and add mask category test cases)
=======
    # TODO add test to verify binary and multiclass output are correct and predictable

=======
>>>>>>> 3db5d22 (Separate selection from extraction)
    def test_data_selector_can_select_images_of_size_512(self):
        selected_path = os.path.join(OUTPUT_PATH, "selected_tiles_512_10_5_42")
        if os.path.exists(selected_path):
            shutil.rmtree(selected_path)
        self.selector.select_data(
            512,
            10,
            5,
            output_path=OUTPUT_PATH,
            random_seed=42,
            lossy=True,
        )
        images_selected = glob.glob(
            os.path.join(selected_path, "**", "*.png"),
            recursive=True,
        )
        self.assertEqual(len(images_selected), (10 + 5) * 2)
>>>>>>> c3118cb (Allow lossy selection of arbitrary image sizes)

    def test_data_selector_picks_different_images_for_different_random_seeds(self):

>>>>>>> 5acb817 (Refactor data selector tests)
=======
    def test_data_selector_picks_different_images_for_different_random_seeds(self):
>>>>>>> 54fa04d (Refactor: separate extraction and selection)
        random_seeds = (43, 44)
        tile_size = 250
        train_n = 10
        test_n = 5

        selected_images = []
<<<<<<< HEAD
<<<<<<< HEAD
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

=======

=======
>>>>>>> 70307a8 (Clean up tests)
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
<<<<<<< HEAD
>>>>>>> 5acb817 (Refactor data selector tests)
=======

>>>>>>> 54fa04d (Refactor: separate extraction and selection)
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
<<<<<<< HEAD
<<<<<<< HEAD
            for test_file in test_fns:
                self.assertNotIn(test_file, train_fns)

    # TODO add test for file specified as output path

=======

=======
>>>>>>> 70307a8 (Clean up tests)
            for test_file in test_fns:
                self.assertNotIn(test_file, train_fns)

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 5acb817 (Refactor data selector tests)
=======
=======
    # TODO add test for file specified as output path
>>>>>>> 3173269 (Add documentation to selection)

>>>>>>> 54fa04d (Refactor: separate extraction and selection)
if __name__ == "__main__":
    unittest.main()
