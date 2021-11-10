from dataloader import DataLoader
import unittest


class TestDataLoader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = "data/curated_1"
        self.dataloader = DataLoader(self.path)

    def test_dataloader_returns_tfdataset_of_correct_shape(self):
        dataset = self.dataloader.to_dataset(n_imgs=10)

        self.assertListEqual(list(dataset.element_spec[0].shape), [224, 224, 3])
        self.assertListEqual(list(dataset.element_spec[1].shape), [224, 224, 1])
