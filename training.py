# For saving models, refere to
# https://www.tensorflow.org/guide/keras/save_and_serialize
import tensorflow

from dataloader import DataLoader
from class_unet_resnet101v2 import Unet
from logging_class import Logs

tensorflow.keras.backend.clear_session()

# parameters of the model.
output_classes = 1  # number of categorical classes.
input_shape = (512, 512, 3)  # input size

batch_size = 4  # batchsize
# Path to the data
path_train = "data/small_large/train"
path_test = "data/small_large/test"


# calling the model.
model = Unet(
    output_classes=output_classes,
    input_shape=input_shape,
    drop_out=True,
    drop_out_rate={"512": 0.275, "256": 0.3, "128": 0.325, "64": 0.35},
)


# Loading the data
dl_train = DataLoader(
    path_train,
    batch_size=batch_size,
    input_shape=input_shape,
    legacy_mode=False,
)

dl_test = DataLoader(
    path_test,
    batch_size=batch_size,
    input_shape=input_shape,
    legacy_mode=False,
)


dl_train.load()
dl_test.load()

train_batches = dl_train.dataset
test_batches = dl_test.dataset

print("data loaded")

# Starting the logs

log = Logs()
comment = "Full large dataset, increasing dropping rate in upstack,\n\
     learning rate divided by two to 0.0005. "
log.main_log(
    comment=comment,
    model_config=model.get_config(),
)

log.local_log(
    train_data_config=dl_train.get_config(),
    val_data_config=dl_test.get_config(),
)

print("log created")
# Preparing the model to be saved using a checkpoint

model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
    filepath=log.checkpoint_filepath,
    save_weights_only=False,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

# Prepare the tensorboard

tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
    log_dir=log.tensorboard_path,
    histogram_freq=1,
    write_graph=True,
)

patience = 30
# Parameters for early stopping
early_stopping = tensorflow.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=patience,
)


print("callbacks defined")

# compiling the model
learning_rate = 0.0005
opt = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
# loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=False,
#     name="sparse_categorical_crossentropy",
# )

loss = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(
    optimizer=opt,
    loss=loss,
    metrics=[
        "accuracy",
        tensorflow.keras.metrics.Recall(name="recall"),
        tensorflow.keras.metrics.Precision(name="precision"),
    ],
)


print("compiling done")
# training the model.
epochs = 1
steps_per_epoch = dl_train.n_samples / batch_size
validation_steps = max(dl_test.n_samples // batch_size, 1)

history = model.fit(
    train_batches,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    validation_data=test_batches,
    callbacks=[
        model_checkpoint_callback,
        tensorboard_callback,
        early_stopping,
    ],
)


# print("first fitting round")
# tensorflow.keras.backend.clear_session()
# print("Starting fine tuning")

# model_dict = model.get_config()

# model_dict["fine_tune_at"] = 4
# model_dict["upstack_trainable"] = True

# # Second you create the model from the configuration dictionary. This creates a new models with the same layer configuration
# best_model = Unet.from_config(model_dict)
# # Finally, you load the weights in the new models.
# # best_model.load_weights(log.checkpoint_filepath)

# opt = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate / 10)

# best_model._down_stack.compile(
#     optimizer=opt,
#     loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=False),
#     metrics=[
#         "accuracy",
#         tensorflow.keras.metrics.Recall(name="recall"),
#         tensorflow.keras.metrics.Precision(name="precision"),
#     ],
# )


# history = best_model._down_stack.fit(
#     train_batches,
#     epochs=epochs,
#     steps_per_epoch=steps_per_epoch,
#     validation_steps=validation_steps,
#     validation_data=test_batches,
#     callbacks=[
#         model_checkpoint_callback,
#         tensorboard_callback,
#         early_stopping,
#     ],
# )

# # I am explaining how to load the model's weight for a custom model.
# # First you get the overwritten configuration of the custim model.
# model_dict = best_model.get_config()

# # Second you create the model from the configuration dictionary. This creates a new models with the same layer configuration
# best_best_model = Unet.from_config(model_dict)
# # Finally, you load the weights in the new models.
# best_best_model.load_weights(log.checkpoint_filepath)


# # logging examples of prediction on the test data sets.
num_batches = 20  # number of batches used to display sample predictions.

log.show_predictions(
    dataset=dl_test.dataset,
    model=model,
    num_batches=num_batches,
)


tensorflow.keras.backend.clear_session()
