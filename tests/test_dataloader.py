import unittest
import math
import os
import numpy as np
from PIL import Image
<<<<<<< HEAD
=======

<<<<<<< HEAD
from dataloader import DataLoader
<<<<<<< HEAD
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======
from dataloader import LegacyModeError
from dataloader import InsuffientDataError
from dataloader import InvalidPathError
>>>>>>> b0c8908 (Use custom exception names)
=======
from loading.dataloader import DataLoader
from loading.dataloader import LegacyModeError
from loading.dataloader import InsuffientDataError
from loading.dataloader import InvalidPathError
>>>>>>> b8e175c (Rename subfolders)

from roof.dataloader import DataLoader
from roof.errors import (
    LegacyModeError,
    InsuffientDataError,
    InvalidPathError,
)


class TestDataLoader(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_paths = [
<<<<<<< HEAD
<<<<<<< HEAD
            os.path.join(
                "data", "testing", "selected_test", "selected_tiles_224", "train"
            ),
            os.path.join(
                "data",
                "testing",
                "selected_test",
                "selected_tiles_500_10_5_42",
                "train",
            ),
        ]

    def _tile_size_from_path(self, data_path):
        folder = os.path.split(os.path.split(data_path)[0])[-1]
        tile_size = int(folder.split("_")[2])
        return tile_size

    def _count_batches(self, dataloader):
        n_elements = dataloader._dataset_input.cardinality().numpy()
        return math.ceil(n_elements / dataloader.batch_size)

    def test_dataloader_returns_tfdataset_of_correct_shape(self):
        for data_path in self.data_paths:
            tile_size = self._tile_size_from_path(data_path)
            if tile_size == 224:
                legacy_mode = True
            else:
                legacy_mode = False

            if tile_size == 224:
                legacy_mode = True
            else:
                legacy_mode = False

            dataloader = DataLoader(
                data_path,
                input_shape=(tile_size, tile_size, 3),
                legacy_mode=legacy_mode,
            )
            dataloader.load()

            n_batches = self._count_batches(dataloader)
=======
            os.path.join("data", "testing", "selected",
                         "selected_tiles_224_fixed", "test_curated_1_final"),
            os.path.join("data", "testing", "selected",
                         "selected_tiles_500_10_5_42_fixed", "train"),
=======
            os.path.join("data", "testing", "selected_test",
                         "selected_tiles_224", "test_curated_1_final"),
            os.path.join("data", "testing", "selected_test",
                         "selected_tiles_500_10_5_42", "train"),
>>>>>>> 54fa04d (Refactor: separate extraction and selection)
        ]

    def _tile_size_from_path(self, data_path):
        folder = os.path.split(os.path.split(data_path)[0])[-1]
        tile_size = int(folder.split("_")[2])
        return tile_size

    def test_dataloader_returns_tfdataset_of_correct_shape(self):
        for data_path in self.data_paths:

            tile_size = self._tile_size_from_path(data_path)

            if tile_size == 224:
                legacy_mode = True
            else:
                legacy_mode = False

            dataloader = DataLoader(
                data_path,
                input_shape=(tile_size, tile_size, 3),
                legacy_mode=legacy_mode,
            )
            dataloader.load()

            # find the number of elements in the tensorflow dataset
            n_batches = math.ceil(dataloader._dataset_input.cardinality(
            ).numpy() / dataloader.batch_size)

            # length = 2
>>>>>>> 4241abc (First working version of data selector with multiclass)
            for inputs, targets in dataloader.dataset.take(n_batches):
                self.assertEqual(inputs.shape, (32, tile_size, tile_size, 3))
                self.assertEqual(targets.shape, (32, tile_size, tile_size, 1))

<<<<<<< HEAD
<<<<<<< HEAD
    def test_dataloader_returns_all_images(self):
        # this test ist done only on the origial curated dataset
        data_path = self.data_paths[0]
        tile_size = self._tile_size_from_path(data_path)
        true_samples = 10
        dataloader = DataLoader(data_path, input_shape=(tile_size, tile_size, 3))
        dataloader.load()
        self.assertEqual(dataloader.n_samples, true_samples)

    def test_dataloader_returns_matching_pairs_map_mask(self):
        for data_path in self.data_paths:
            tile_size = self._tile_size_from_path(data_path)
            if tile_size == 224:
                legacy_mode = True
            else:
                legacy_mode = False

            dataloader = DataLoader(
                data_path,
                input_shape=(tile_size, tile_size, 3),
                legacy_mode=legacy_mode,
            )

            n_batches = self._count_batches(dataloader)
            map_paths = list(dataloader._dataset_input.take(n_batches))
            mask_paths = list(dataloader._dataset_target.take(n_batches))

            for map_path, mask_path in zip(map_paths, mask_paths):
                map_name = map_path.numpy().decode("utf-8").split("map")[0]
                if "mask" in mask_path.numpy().decode("utf-8"):
                    mask_name = mask_path.numpy().decode("utf-8").split("mask")[0]
                else:
                    mask_name = mask_path.numpy().decode("utf-8").split("msk")[0]

                self.assertEqual(map_name, mask_name)

    def test_dataloader_returns_mask_with_expected_range_of_values_binary(self):
        for data_path in self.data_paths:
            tile_size = self._tile_size_from_path(data_path)
            if tile_size == 224:
                legacy_mode = True
            else:
                legacy_mode = False
            dataloader = DataLoader(
                data_path,
                input_shape=(tile_size, tile_size, 3),
                legacy_mode=legacy_mode,
            )
            dataloader.load()

            n_batches = self._count_batches(dataloader)
            mask_set = set([])
            for _, targets in dataloader.dataset.take(n_batches):
                for target in targets:
                    mask_set.update(list(target.numpy().flatten()))
            true_set = {0, 1}
            self.assertSetEqual(mask_set, true_set)

    def test_dataloader_returns_mask_with_expected_range_of_values_multiclass(self):
        # this test can not be done on the original curated dataset (legacy_mode=True)
        for data_path in self.data_paths[1:]:
            tile_size = self._tile_size_from_path(data_path)
            legacy_mode = False
            dataloader = DataLoader(
                data_path,
                input_shape=(tile_size, tile_size, 3),
                legacy_mode=legacy_mode,
                multiclass=True,
            )
            dataloader.load()

            n_batches = self._count_batches(dataloader)
            mask_set = set([])
            for _, targets in dataloader.dataset.take(n_batches):
                for target in targets:
                    mask_set.update(list(target.numpy().flatten()))

            true_set = {0.0, 1.0, 2.0, 3.0, 4.0}
            print(mask_set)
            print(true_set)
            self.assertSetEqual(mask_set, true_set)

    def test_dataloader_does_not_permit_legacy_multiclass(self):
        data_path = self.data_paths[0]
        with self.assertRaises(AssertionError):
            DataLoader(data_path, legacy_mode=True, multiclass=True)

    def test_dataloader_raises_error_invalid_path(self):
        with self.assertRaises(InvalidPathError):
            DataLoader("/invalid/path")

    def test_dataloader_raises_error_no_valid_images_found(self):
        tile_size = 600  # non-matching tile size
        for data_path in self.data_paths:
            with self.assertRaises(InsuffientDataError):
                DataLoader(data_path, input_shape=(tile_size, tile_size, 3))

    def test_dataloader_raises_error_legacy_mode_new_data(self):
        tile_size = 500
        with self.assertRaises(LegacyModeError):
            DataLoader(
                self.data_paths[1],  # new type dataset
                input_shape=(tile_size, tile_size, 3),
                legacy_mode=True,  # legacy mode should not be allowed
            )

    def test_dataloader_returns_expected_values_for_binary_mode_more(self):
        image_dir = "data/testing/selected_test/selected_tiles_250_10_5_42/more_roof/"
        dataloader = DataLoader(
            image_dir,
            input_shape=(250, 250, 3),
            legacy_mode=False,
            multiclass=False,
            batch_size=1,
        )
        dataloader.load()

        for image_fn in os.listdir(image_dir):
            if "msk" in image_fn:
                msk_fn = os.path.join(image_dir, image_fn)
                msk = np.array(Image.open(msk_fn))
                true_roof = (msk > 0).sum()

        for _, msk in dataloader.dataset.take(1):
            roof = msk.numpy() > 0
            no_roof = msk.numpy() == 0
            self.assertGreater(np.sum(roof), 0)
            self.assertLess(np.sum(no_roof), np.sum(roof))
            self.assertEqual(np.sum(roof), true_roof)

    def test_dataloader_returns_expected_values_for_binary_mode_less(self):
        image_dir = "data/testing/selected_test/selected_tiles_250_10_5_42/less_roof/"
        dataloader = DataLoader(
            image_dir,
            input_shape=(250, 250, 3),
            legacy_mode=False,
            multiclass=False,
            batch_size=1,
        )
        dataloader.load()

        for image_fn in os.listdir(image_dir):
            if "msk" in image_fn:
                msk_fn = os.path.join(image_dir, image_fn)
                msk = np.array(Image.open(msk_fn))
                true_roof = (msk > 0).sum()

        for _, msk in dataloader.dataset.take(1):
            roof = msk.numpy() > 0
            no_roof = msk.numpy() == 0
            self.assertGreater(np.sum(roof), 0)
            self.assertGreater(np.sum(no_roof), np.sum(roof))
            self.assertEqual(np.sum(roof), true_roof)

    def test_dataloader_returns_expected_values_for_multiclass_mode(self):
        image_dirs = [
            "data/testing/selected_test/selected_tiles_250_10_5_42/pv3/",
            "data/testing/selected_test/selected_tiles_250_10_5_42/pv4/",
        ]

        for image_dir in image_dirs:
            dataloader = DataLoader(
                image_dir,
                input_shape=(250, 250, 3),
                legacy_mode=False,
                multiclass=True,
                batch_size=1,
            )
            dataloader.load()

            for image_fn in os.listdir(image_dir):
                if "msk" in image_fn:
                    msk_fn = os.path.join(image_dir, image_fn)
                    msk = np.array(Image.open(msk_fn))
                    true_no_roof = (msk == 0).sum()
                    true_pv1 = (msk == 63).sum()
                    true_pv2 = (msk == 127).sum()
                    true_pv3 = (msk == 191).sum()
                    true_pv4 = (msk == 255).sum()

            true_values = [true_no_roof, true_pv1, true_pv2, true_pv3, true_pv4]

            for _, msk in dataloader.dataset.take(1):
                no_roof = (msk.numpy() == 0).sum()
                pv1 = (msk.numpy() == 1).sum()
                pv2 = (msk.numpy() == 2).sum()
                pv3 = (msk.numpy() == 3).sum()
                pv4 = (msk.numpy() == 4).sum()
                self.assertListEqual([no_roof, pv1, pv2, pv3, pv4], true_values)
=======
    def test_dataloader_returns_requested_number_of_images(self):

        for data_path in self.data_paths:
            tile_size = int(data_path.split("_")[2])
            n_samples = 10
            dataloader = DataLoader(
                data_path, n_samples=n_samples, input_shape=(tile_size, tile_size, 3))
            dataloader.load()
            self.assertEqual(
                dataloader._dataset_input.cardinality().numpy(), n_samples)

    def test_dataloader_raises_assert_error_too_many_images_requested(self):
        for data_path in self.data_paths:
            tile_size = int(data_path.split("_")[2])
            n_samples = 10
            with self.assertRaises(AssertionError):
                DataLoader(data_path, n_samples=1_000_000_000)

    def test_dataloader_does_not_fail_without_n_samples_set(self):
=======
    def test_dataloader_returns_all_images(self):
>>>>>>> c8ec9a0 (Remove n_samples from dataloader and add some error handling)
        # this test ist done only on the origial curated dataset
        data_path = self.data_paths[0]
        tile_size = self._tile_size_from_path(data_path)
        true_samples = 256
        dataloader = DataLoader(
            data_path, input_shape=(tile_size, tile_size, 3))
        dataloader.load()
        self.assertEqual(dataloader.n_samples, true_samples)

    def test_dataloader_returns_matching_pairs_map_mask(self):
        for data_path in self.data_paths:
            tile_size = self._tile_size_from_path(data_path)

            if tile_size == 224:
                legacy_mode = True
            else:
                legacy_mode = False

            dataloader = DataLoader(
                data_path,
                input_shape=(tile_size, tile_size, 3),
                legacy_mode=legacy_mode,
                )

            # find the number of elements in the tensorflow dataset
            n_batches = math.ceil(dataloader._dataset_input
                                  .cardinality().numpy() / dataloader.batch_size)

            map_paths = list(
                dataloader._dataset_input.take(n_batches))
            mask_paths = list(
                dataloader._dataset_target.take(n_batches))

            for map_path, mask_path in zip(map_paths, mask_paths):
                map_name = map_path.numpy().decode("utf-8").split("map")[0]
                if "mask" in mask_path.numpy().decode(
                        "utf-8"):
                    mask_name = mask_path.numpy().decode(
                        "utf-8").split("mask")[0]
                else:
                    mask_name = mask_path.numpy().decode(
                        "utf-8").split("msk")[0]

                self.assertEqual(map_name, mask_name)

    def test_dataloader_returns_mask_with_expected_range_of_values_binary(self):
        for data_path in self.data_paths:
            tile_size = self._tile_size_from_path(data_path)
            if tile_size == 224:
                legacy_mode = True
            else:
                legacy_mode = False
            dataloader = DataLoader(
                data_path,
                input_shape=(tile_size, tile_size, 3),
                legacy_mode=legacy_mode
                )
            dataloader.load()
            # find the number of elements in the tensorflow dataset
            n_batches = math.ceil(dataloader._dataset_input
                                  .cardinality().numpy() / dataloader.batch_size)

            mask_set = set([])
            for _, targets in dataloader.dataset.take(n_batches):
                for target in targets:
                    mask_set.update(list(target.numpy().flatten()))
            true_set = {0, 1}
            self.assertSetEqual(mask_set, true_set)

    def test_dataloader_returns_mask_with_expected_range_of_values_multiclass(self):
        # this test can not be done on the original curated dataset (legacy_mode=True)
        for data_path in self.data_paths[1:]:
            tile_size = self._tile_size_from_path(data_path)
            legacy_mode = False
            dataloader = DataLoader(
                data_path,
                input_shape=(tile_size, tile_size, 3),
                legacy_mode=legacy_mode, multiclass=True
                )
            dataloader.load()
            # find the number of elements in the tensorflow dataset
            n_batches = math.ceil(dataloader._dataset_input
                                  .cardinality().numpy() / dataloader.batch_size)

            mask_set = set([])
            for _, targets in dataloader.dataset.take(n_batches):
                for target in targets:
                    mask_set.update(list(target.numpy().flatten()))

            true_set = {0.0, 1.0, 2.0, 3.0, 4.0}
            print(mask_set)
            print(true_set)
            self.assertSetEqual(mask_set, true_set)

    def test_dataloader_does_not_permit_legacy_multiclass(self):
        data_path = self.data_paths[0]
        with self.assertRaises(AssertionError):
            DataLoader(data_path, legacy_mode=True, multiclass=True)
<<<<<<< HEAD
>>>>>>> 4241abc (First working version of data selector with multiclass)
=======

    def test_dataloader_raises_error_invalid_path(self):
        with self.assertRaises(InvalidPathError):
            DataLoader("/invalid/path")

    def test_dataloader_raises_error_no_valid_images_found(self):
        tile_size = 600  # non-matching tile size
        for data_path in self.data_paths:
            with self.assertRaises(InsuffientDataError):
                DataLoader(
                    data_path,
                    input_shape=(tile_size, tile_size,3)
                )
<<<<<<< HEAD
>>>>>>> c8ec9a0 (Remove n_samples from dataloader and add some error handling)
=======

    def test_dataloader_raises_error_legacy_mode_new_data(self):
        tile_size = 500
        with self.assertRaises(LegacyModeError):
            DataLoader(
                self.data_paths[1],  # new type dataset
                input_shape=(tile_size, tile_size, 3),
                legacy_mode=True  # legacy mode should not be allowed
            )
<<<<<<< HEAD
>>>>>>> a887e5d (Add legacy mode sanity check (by filename))
=======

    def test_dataloader_returns_expected_values_for_binary_mode_more(self):
        image_dir = "data/testing/selected_test/selected_tiles_250_10_5_42/more_roof/"
        dataloader = DataLoader(
            image_dir,
            input_shape=(250, 250, 3),
            legacy_mode=False,
            multiclass=False,
            batch_size=1,
        )
        dataloader.load()

        for image_fn in os.listdir(image_dir):
            if "msk" in image_fn:
                msk_fn = os.path.join(image_dir, image_fn)
                msk = np.array(Image.open(msk_fn))
                true_roof = (msk > 0).sum()

        for _, msk in dataloader.dataset.take(1):
            roof = msk.numpy() > 0
            no_roof = msk.numpy() == 0
            self.assertGreater(np.sum(roof), 0)
            self.assertLess(np.sum(no_roof), np.sum(roof))
            self.assertEqual(np.sum(roof), true_roof)

    def test_dataloader_returns_expected_values_for_binary_mode_less(self):

        image_dir = "data/testing/selected_test/selected_tiles_250_10_5_42/less_roof/"
        dataloader = DataLoader(
            image_dir,
            input_shape=(250, 250, 3),
            legacy_mode=False,
            multiclass=False,
            batch_size=1,
        )
        dataloader.load()

        for image_fn in os.listdir(image_dir):
            if "msk" in image_fn:
                msk_fn = os.path.join(image_dir, image_fn)
                msk = np.array(Image.open(msk_fn))
                true_roof = (msk > 0).sum()

        for _, msk in dataloader.dataset.take(1):
            roof = msk.numpy() > 0
            no_roof = msk.numpy() == 0
            self.assertGreater(np.sum(roof), 0)
            self.assertGreater(np.sum(no_roof), np.sum(roof))
            self.assertEqual(np.sum(roof), true_roof)

    def test_dataloader_returns_expected_values_for_multiclass_mode(self):
        image_dirs = [
            "data/testing/selected_test/selected_tiles_250_10_5_42/pv3/",
            "data/testing/selected_test/selected_tiles_250_10_5_42/pv4/",
        ]

        for image_dir in image_dirs:
            dataloader = DataLoader(
                image_dir,
                input_shape=(250, 250, 3),
                legacy_mode=False,
                multiclass=True,
                batch_size=1,
            )
            dataloader.load()

            for image_fn in os.listdir(image_dir):
                if "msk" in image_fn:
                    msk_fn = os.path.join(image_dir, image_fn)
                    msk = np.array(Image.open(msk_fn))
                    true_no_roof = (msk == 0).sum()
                    true_pv1 = (msk == 63).sum()
                    true_pv2 = (msk == 127).sum()
                    true_pv3 = (msk == 191).sum()
                    true_pv4 = (msk == 255).sum()

            true_values = [true_no_roof, true_pv1,
                           true_pv2, true_pv3, true_pv4]

            for _, msk in dataloader.dataset.take(1):
                no_roof = (msk.numpy() == 0).sum()
                pv1 = (msk.numpy() == 1).sum()
                pv2 = (msk.numpy() == 2).sum()
                pv3 = (msk.numpy() == 3).sum()
                pv4 = (msk.numpy() == 4).sum()
                self.assertListEqual(
                    [no_roof, pv1, pv2, pv3, pv4],
                    true_values
                )
>>>>>>> 1658920 (Fix binary mask loading and add mask category test cases)
