""" This test file check the loading of the dataloader and the model.
"""
import numpy as np
import tensorflow as tf
from unet.unet_resnet101v2_pt import Model

PATH_TRAIN = "data/small_large/train"
PATH_VAL = "data/small_large/test"

# confiuguration for Resnet
layer_names = [
    "conv1_conv",
    "conv2_block2_out",
    "conv3_block3_out",
    "conv4_block22_out",
    "conv5_block2_out",
]


model_name = "Resnet101v2"


tf.keras.backend.clear_session()

alpha = 1.0
epochs = 3
fine_tune_at = 5
fine_tune_epoch = 3
patience = 7
drop_out_rate = {"512": 0.2, "256": 0.25, "128": 0.3, "64": 0.35}


model = Model(
    path_train=PATH_TRAIN,
    path_test=PATH_VAL,
    layer_names=layer_names,
    output_classes=1,
    input_shape=(512, 512, 3),
    epochs=epochs,
    fine_tune_epoch=fine_tune_epoch,
    batch_size=4,
    model_name=model_name,
    include_top=False,
    pooling=max,
    fine_tune_at=fine_tune_at,
    drop_out=True,
    drop_out_rate=drop_out_rate,
    patience=patience,
)
comment = f" Test run. Unet model, {model_name}, weight on imagenet turned on,\n\
         base model not trainable, new layers for concatenation following the central line\n\
         fine tuning deactivated, dropout rate: {drop_out_rate}.\n\
        Large pictures, full sample."

model_history = model.model_history(comment)

print("Done")


tf.keras.backend.clear_session()
