import unittest
import math
import os
import numpy as np
from PIL import Image

from dataloader import DataLoader


class TestDataLoader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_paths = [
            os.path.join("data", "testing", "selected",
                         "selected_tiles_224_fixed", "test_curated_1_final"),
            os.path.join("data", "testing", "selected",
                         "selected_tiles_500_10_5_42_fixed", "train"),
        ]

    def test_dataloader_returns_tfdataset_of_correct_shape(self):
        for data_path in self.data_paths:

            tile_size = int(data_path.split("_")[2])

            dataloader = DataLoader(
                data_path,
                input_shape=(tile_size, tile_size, 3)
            )
            dataloader.load()

            # find the number of elements in the tensorflow dataset
            n_batches = math.ceil(dataloader._dataset_input.cardinality(
            ).numpy() / dataloader.batch_size)

            # length = 2
            for inputs, targets in dataloader.dataset.take(n_batches):
                self.assertEqual(inputs.shape, (32, tile_size, tile_size, 3))
                self.assertEqual(targets.shape, (32, tile_size, tile_size, 1))

    def test_dataloader_returns_all_images(self):
        # this test ist done only on the origial curated dataset
        data_path = self.data_paths[0]
        tile_size = int(data_path.split("_")[2])
        true_samples = 256
        dataloader = DataLoader(
            data_path, input_shape=(tile_size, tile_size, 3))
        dataloader.load()
        self.assertEqual(dataloader.n_samples, true_samples)

    def test_dataloader_returns_matching_pairs_map_mask(self):
        for data_path in self.data_paths:
            tile_size = int(data_path.split("_")[2])
            dataloader = DataLoader(
                data_path,
                input_shape=(tile_size, tile_size, 3)
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
            tile_size = int(data_path.split("_")[2])
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
            tile_size = int(data_path.split("_")[2])
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

    def test_dataloader_raises_error_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            DataLoader("/invalid/path")

    def test_dataload_raises_error_no_valid_images_found(self):

        tile_size = 600  # non-matching tile size

        for data_path in self.data_paths:
            with self.assertRaises(FileNotFoundError):
                DataLoader(
                    data_path,
                    input_shape=(tile_size, tile_size,3)
                )
