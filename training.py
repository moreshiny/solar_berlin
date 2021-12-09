# For saving models, refere to
# https://www.tensorflow.org/guide/keras/save_and_serialize
import tensorflow

# import tensorflow_addons as tfa
from roof.dataloader import DataLoader
from class_unet_resnet101v2 import Unet
from logging_class import Logs

tensorflow.keras.backend.clear_session()

# parameters of the model.
output_classes = 5  # number of categorical classes.
input_shape = (512, 512, 3)  # input size
epochs = 3

batch_size = 8  # batchsize
# Path to the data large multiclass dataset
# path_train = "data/selected_tiles_512_4000_1000_42_partial/train"
# path_test = "data/selected_tiles_512_4000_1000_42_partial/test"

# Path to the data small multiclass dataset
path_train = "data/selected_512_multiclass/selected_tiles_512_100_20_42/train"
path_test = "data/selected_512_multiclass/selected_tiles_512_100_20_42/test"

# path to the small mono class large dataset
# path_train = "data/small_large/train"
# path_test = "data/small_large/test"

# calling the model.
model = Unet(
    output_classes=output_classes,
    input_shape=input_shape,
    drop_out=True,
    drop_out_rate={"512": 0.275, "256": 0.3, "128": 0.325, "64": 0.35},
    multiclass=bool(output_classes - 1),
)

# Starting the logs

log = Logs()
comment = "Full large dataset, multiclassification problem ,\n\
     standard learning rate.  "
log.main_log(
    comment=comment,
    model_config=model.get_config(),
)


print("log created")


# listing the metrics which are used in the model.

binary_accuracy = tensorflow.keras.metrics.BinaryAccuracy(name="accuracy")
sparse_categorical_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(
    name="sparse_categorical_accuracy", dtype=None
)
recall = tensorflow.keras.metrics.Recall(name="recall")
precision = tensorflow.keras.metrics.Precision(name="precision")

if output_classes > 1:
    multiclass = True
    loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metric_list = [
        "sparse_categorical_accuracy",
    ]
    metrics = [sparse_categorical_accuracy]
    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=log.checkpoint_filepath,
        save_weights_only=False,
        monitor="sparse_categorical_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
else:
    multiclass = False
    loss = tensorflow.keras.losses.BinaryCrossentropy(from_logits=False)
    metric_list = [
        "accuracy",
        "recall",
        "precision",
    ]
    metrics = [binary_accuracy, precision, recall]
    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=log.checkpoint_filepath,
        save_weights_only=False,
        monitor="accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

dl_train = DataLoader(
    path_train,
    batch_size=batch_size,
    input_shape=input_shape,
    legacy_mode=False,
    multiclass=multiclass,
)


dl_test = DataLoader(
    path_test,
    batch_size=batch_size,
    input_shape=input_shape,
    legacy_mode=False,
    multiclass=multiclass,
)


dl_train.load()
dl_test.load()

train_batches = dl_train.dataset
test_batches = dl_test.dataset


print("data loaded")


# Preparing the model to be saved using a checkpoint


# Prepare the tensorboard

tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
    log_dir=log.tensorboard_path,
    histogram_freq=1,
    write_graph=True,
)

patience = 7
# Parameters for early stopping
early_stopping = tensorflow.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=patience,
)


print("callbacks defined")

# compiling the model
learning_rate = 0.0001
opt = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)


model.compile(optimizer=opt, loss=loss, metrics=metrics)


print("compiling done")
# training the model.

steps_per_epoch = dl_train.n_samples / batch_size
validation_steps = max(dl_test.n_samples // batch_size, 1)


history = model.fit(
    train_batches,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    validation_data=test_batches,
    callbacks=[
        # model_checkpoint_callback,
        # tensorboard_callback,
        early_stopping,
    ],
)

accuracies = {}
accuracies["loss"] = [history.history["loss"], history.history["val_" + "loss"]]

for metric in metric_list:
    accuracies[metric] = [history.history[metric], history.history["val_" + metric]]

print(accuracies)

log.local_log(
    train_data_config=dl_train.get_config(),
    val_data_config=dl_test.get_config(),
    metrics=accuracies,
)

num_batches = 3  # number of batches
log.show_predictions(
    dataset=dl_test.dataset,
    model=model,
    num_batches=num_batches,
    multiclass=multiclass,
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
# number of batches used to display sample predictions.


tensorflow.keras.backend.clear_session()
