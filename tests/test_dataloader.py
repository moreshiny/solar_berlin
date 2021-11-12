import math
from dataloader import DataLoader
import unittest


class TestDataLoader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = "data/curated_1/test_curated_1_final/"
        self.dataloader = DataLoader(self.path)

    def test_dataloader_returns_tfdataset_of_correct_shape(self):
        self.dataloader.load()

        # find the number of elements in the tensorflow dataset
        length = math.ceil(self.dataloader._dataset_input.cardinality(
        ).numpy() / self.dataloader.batch_size)

        # length = 2
        for x, y in self.dataloader.dataset.take(length):
            self.assertEqual(x.shape, (32, 224, 224, 3))
            self.assertEqual(y.shape, (32, 224, 224))
