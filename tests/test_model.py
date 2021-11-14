import unittest

from model import Model


class TestModel(unittest.TestCase):

    def test_model_simple_run_does_not_result_in_error(self):

        train_path = "data/test_selected/train"
        test_path = "data/test_selected/test"

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
            epochs=2,
        )

        _ = model.model_history()
