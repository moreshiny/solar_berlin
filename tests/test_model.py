import unittest
import os

from unet.unet_basic_nt import Model


class TestModel(unittest.TestCase):

    def test_model_simple_run_does_not_result_in_error(self):
        train_path = os.path.join(
            "data", "testing", "selected_test", "selected_tiles_224", "train"
        )
        test_path = os.path.join(
            "data", "testing", "selected_test", "selected_tiles_224", "test"
        )

        layer_names = [
            "block_1_expand_relu",   # 64x64
            "block_3_expand_relu",   # 32x32
            "block_6_expand_relu",   # 16x16
            "block_13_expand_relu",  # 8x8
            "block_16_project",      # 4x4
        ]

        model = Model(
            train_path,
            test_path,
            layer_names,
            epochs=1,
            batch_size=8
        )

        _ = model.model_history()