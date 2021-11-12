import unittest
import math
from dataloader import DataLoader


class TestDataLoader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = "data/curated_1/test_curated_1_final/"

    def test_dataloader_returns_tfdataset_of_correct_shape(self):
        dataloader = DataLoader(self.path, n_samples=10)
        dataloader.load()

        # find the number of elements in the tensorflow dataset
        n_batches = math.ceil(dataloader._dataset_input.cardinality(
        ).numpy() / dataloader.batch_size)

        # length = 2
        for inputs, targets in dataloader.dataset.take(n_batches):
            self.assertEqual(inputs.shape, (32, 224, 224, 3))
            self.assertEqual(targets.shape, (32, 224, 224))

    def test_dataloader_returns_requested_number_of_images(self):
        dataloader = DataLoader(self.path, n_samples=10)
        dataloader.load()
        self.assertEqual(dataloader._dataset_input.cardinality().numpy(), 10)

    def test_dataloader_raises_assert_error_too_many_images_requested(self):
        with self.assertRaises(AssertionError):
            DataLoader(self.path, n_samples=1_000_000_000)

    def test_dataloader_does_not_fail_withou_n_samples_set(self):
        dataloader = DataLoader(self.path)
        dataloader.load()
        self.assertEqual(dataloader.n_samples, 256)

    def test_dataloader_returns_matching_pairs_map_mask(self):
        dataloader = DataLoader(self.path, n_samples=100)

        # find the number of elements in the tensorflow dataset
        n_batches = math.ceil(dataloader._dataset_input
                              .cardinality().numpy() / dataloader.batch_size)

        map_paths = list(dataloader._dataset_input.take(n_batches))
        mask_paths = list(dataloader._dataset_target.take(n_batches))

        for map_path, mask_path in zip(map_paths, mask_paths):
            self.assertEqual(
                map_path.numpy().decode("utf-8").split("map"),
                mask_path.numpy().decode("utf-8").split("mask")
            )
