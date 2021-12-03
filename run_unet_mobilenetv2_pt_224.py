import tensorflow
from models.unet_mobilenetv2_pt import Model

PATH_TRAIN = "data/test_data_224/train"
PATH_VAL = "data/test_data_224/test"

layer_names = [
    "block_1_expand_relu",  # 64x64
    "block_3_expand_relu",  # 32x32
    "block_6_expand_relu",  # 16x16
    "block_13_expand_relu",  # 8x8
    "block_16_project",  # 4x4
]
model_name = "mobilenetv2_pt"
epochs = 1
batch_size = 8
alpha = 1.0

tensorflow.keras.backend.clear_session()

model = Model(
    path_train=PATH_TRAIN,
    path_test=PATH_VAL,
    layer_names=layer_names,
    output_classes=1,
    input_shape=(224, 224, 3),
    epochs=epochs,
    fine_tune_epoch=0,
    batch_size=batch_size,
    model_name=model_name,
    include_top=False,
    alpha=alpha,
    pooling=max,
    fine_tune_at=0,
)

comment = f"Unet model, {model_name}, alpha =1, weight on imagenet turned on,\n\
        base model trainable, none of the layers are trainable. Fine tuning turned off,\
        for the test data 512x512."

model.model_history(comment)

print("Done")
