""" This test file check the loading of the dataloader and the model.
"""
import numpy as np
from model import Model

PATH_TRAIN = "data/curated_split/train"
PATH_VAL = "data/curated_split/val"


layer_names = [
    "block_1_expand_relu",  # 64x64
    "block_3_expand_relu",  # 32x32
    "block_6_expand_relu",  # 16x16
    "block_13_expand_relu",  # 8x8
    "block_16_project",  # 4x4
]


model_name = "mobilenetv2"


alpha = 1.0

for fine_tune_at in range(10, 150, 10):

    model = Model(
        path_train=PATH_TRAIN,
        path_test=PATH_VAL,
        layer_names=layer_names,
        output_classes=1,
        input_shape=(224, 224, 3),
        epochs=8,
        fine_tune_epoch=12,
        batch_size=128,
        model_name=model_name,
        include_top=False,
        alpha=alpha,
        pooling=max,
        fine_tune_at=fine_tune_at,
    )

    comment = f"Unet model, {model_name}, second run, alpha =1, weight on imagenet turned on,\n\
        base model trainable, {fine_tune_at} of the layers are trainable. fine tuning turned on,\
        for the curated set. "

    model_history = model.model_history(comment)
