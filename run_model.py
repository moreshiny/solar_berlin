import tensorflow
from models.unet_resnet101v2_pt import Model

PATH_TRAIN = "data/selected_512/train"
PATH_VAL = "data/selected_512/test"

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
epochs = 30
fine_tune_at = 0
fine_tune_epoch = 0
patience = 10
drop_out_rate = {"512": 0.2, "256": 0.25, "128": 0.3, "64": 0.35}
batch_size = 4
buffer_size = 250



model = Model(
    path_train=PATH_TRAIN,
    path_test=PATH_VAL,
    layer_names=layer_names,
    output_classes=1,
    input_shape=(512, 512, 3),
    epochs=epochs,
    fine_tune_epoch=fine_tune_epoch,
    batch_size=batch_size,
    model_name=model_name,
    include_top=False,
    pooling=max,
    fine_tune_at=fine_tune_at,
    drop_out=True,
    drop_out_rate=drop_out_rate,
    patience=patience,
    buffer_size=buffer_size
)
comment = f"Current Golden Unet model, {model_name}, weight on imagenet turned on,\n\
         base model not trainable, new layers for concatenation following the central line\n\
         fine tuning deactivated, dropout rate: {drop_out_rate}.\n\
        Large pictures, full sample."

model_history = model.model_history(comment)

print("Done")
