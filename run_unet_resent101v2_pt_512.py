""" This test file check the loading of the dataloader and the model.
"""
import numpy as np
import tensorflow
from model import Model

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


tensorflow.keras.backend.clear_session()

alpha = 1.0
epochs = 1
fine_tune_at = 1
fine_tune_epoch = 1
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
)
comment = f"Unet model, {model_name}, weight on imagenet turned on,\n\
         base model not trainable, new layers for concatenation following the central line\n\
         fine tuning deactivated, dropout rate: {drop_out_rate}.\n\
        Large pictures, full sample. Second implementation\n\
        of the early stopping."

model_history = model.model_history(comment)

print("Done")


tensorflow.keras.backend.clear_session()


# testing dropout
# alpha = 1.0
# epochs = 100
# fine_tune_at = 0
# fine_tune_epoch = 0
# drop_out_rates = np.arange(0.1, 0.5, 0.1)

# for drop_out_rate in drop_out_rates:

#     model = Model(
#         path_train=PATH_TRAIN,
#         path_test=PATH_VAL,
#         layer_names=layer_names,
#         output_classes=1,
#         input_shape=(512, 512, 3),
#         epochs=epochs,
#         fine_tune_epoch=fine_tune_epoch,
#         batch_size=16,
#         model_name=model_name,
#         include_top=False,
#         alpha=alpha,
#         pooling=max,
#         fine_tune_at=fine_tune_at,
#         drop_out=True,
#         drop_out_rate=drop_out_rate,
#     )
#     comment = f"Unet model, {model_name}, weight on imagenet turned on,\n\
#             base model not trainable, new layers for concatenation follwing the central line\n\
#             fine tuning deactivated, dropout activated. Large pictures, full sample. First implementation\n\
#             of the early stopping."

#     model_history = model.model_history(comment)

#     print("Done")
#     tensorflow.keras.backend.clear_session()


# # Testing fine tuning
alpha = 1.0
epochs = 100
fine_tune_epoch = 100
fine_tune_layer = range(5, 40, 5)
drop_out_rate = 0.3

for fine_tune_layer in range(5, 40, 5):

    model = Model(
        path_train=PATH_TRAIN,
        path_test=PATH_VAL,
        layer_names=layer_names,
        output_classes=1,
        input_shape=(512, 512, 3),
        epochs=epochs,
        fine_tune_epoch=fine_tune_epoch,
        batch_size=8,
        model_name=model_name,
        include_top=False,
        pooling=max,
        fine_tune_at=fine_tune_layer,
        drop_out=True,
    )
    comment = f"Unet model, {model_name}, weight on imagenet turned on,\n\
            base model not trainable, new layers for concatenation follwing the central line\n\
            fine tuning activated, dropout activated, at the best tested rate.\n\
            Large pictures, full sample. Second implementation of the early stopping.\n\
            Loading the best model from the first run."

    model_history = model.model_history(comment)

    print("Done")
    tensorflow.keras.backend.clear_session()
