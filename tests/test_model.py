import unittest

from dataloader import DataLoader
from model import Model


class TestModel(unittest.TestCase):

    def test_model_runs(self):

        path = "data/curated_1"
        dataloader = DataLoader(path)

        TRAIN_LENGTH = 10
        dataset = dataloader.to_dataset(TRAIN_LENGTH)

        train_size = int(0.8 * TRAIN_LENGTH)
        test_size = int(0.2 * TRAIN_LENGTH)
        dataset_train = dataset.take(train_size)

        dataset_val = dataset.take(test_size)

        layer_names = [
            "block_1_expand_relu",   # 64x64
            "block_3_expand_relu",   # 32x32
            "block_6_expand_relu",   # 16x16
            "block_13_expand_relu",  # 8x8
            "block_16_project",      # 4x4
        ]

        model = Model(
            dataset_train,
            dataset_val,
            layer_names,
            epochs=2,
        )

        history = model.model_history()

        print(history)
