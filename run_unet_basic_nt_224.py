import tensorflow

from models.unet_basic_nt import Model

PATH_TRAIN = "data/test_data_224/train"
PATH_VAL = "data/test_data_224/test"

layer_names = []
model_name = "UNET"
epochs = 1
batch_size = 8

tensorflow.keras.backend.clear_session()

model = Model(
    path_train=PATH_TRAIN,
    path_test=PATH_VAL,
    layer_names=layer_names,
    epochs=epochs,
    model_name=model_name,
    batch_size=batch_size,
)

model.model_history()

print("Done")
