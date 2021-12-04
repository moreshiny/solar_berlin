# For saving models, refere to
# https://www.tensorflow.org/guide/keras/save_and_serialize
import tensorflow
from class_unet_resnet101v2 import Unet
from roof.dataloader import DataLoader


# Layers for the skip connections.
layer_names = [
    "conv1_conv",
    "conv2_block2_out",
    "conv3_block3_out",
    "conv4_block22_out",
    "conv5_block2_out",
]
# parameters of the model.
output_classes = 1  # number of categorical classes.
input_shape = (512, 512, 3)  # input size
batch_size = 4  # batchsize
# Path to the data
path_train = "data/small_large/train"
path_test = "data/small_large/test"


# calling the model.
model = Unet(
    layer_names=layer_names,
    output_classes=output_classes,
    drop_out=True,
    drop_out_rate={"512": 0.25, "256": 0.3, "128": 0.35, "64": 0.4},
    fine_tune_at=20,
)


# Loading the data
dl_train = DataLoader(
    path_train,
    batch_size=batch_size,
    input_shape=input_shape,
)
dl_val = DataLoader(
    path_test,
    batch_size=batch_size,
    input_shape=input_shape,
)
dl_train.load()
dl_val.load()


train_batches = dl_train.dataset
test_batches = dl_val.dataset


# compiling the model
learning_rate = 0.001
opt = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=opt,
    loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        "accuracy",
        tensorflow.keras.metrics.Recall(name="recall"),
        tensorflow.keras.metrics.Precision(name="precision"),
    ],
)


# training the model.
epochs = 1
steps_per_epoch = dl_train.n_samples / batch_size
validation_steps = max(dl_val.n_samples // batch_size, 1)

history = model.fit(
    train_batches,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

model_dict = model.get_config()

print(model_dict)

# saving the model

tensorflow.saved_model.save(model, "mymodel")

# Loading model weights. Note that the full model cannot be loaded for subclassed model
loaded_1 = tensorflow.saved_model.load("mymodel")
