# For saving models, refere to
# https://www.tensorflow.org/guide/keras/save_and_serialize
import numpy as np
import tensorflow

# import tensorflow_addons as tfa

from roof.dataloader import DataLoader
from class_unet_resnet101v2 import Unet
from logging_class import Logs

tensorflow.keras.backend.clear_session()

# parameters of the model.
OUTPUT_CLASSES = 5  # number of categorical classes. for 2 classes = 1.
INPUT_SHAPE = (512, 512, 3)  # input size
EPOCHS = 100
PATIENCE = 20

BATCH_SIZE = 32  # batchsize

NUM_BATCHES = 2  # number of batches for predictions


COMMENT = "Full large dataset ( with the 30pc highst loss cleaned with the \n\
    with the latest model, and the 25pc empty frame discarded, multiclassification\n\
    problem, standard learning rate. "

# Path to data
PATH_TRAIN = "data/cleaned/bin_clean_8000/train_unet/transparent"
PATH_TEST = "data/cleaned/bin_clean_8000/test"


# Path to the data large multiclass dataset
# path_train = "data/selected_tiles_512_4000_1000_42_partial/train"
# path_test = "data/selected_tiles_512_4000_1000_42_partial/test"

# Path to the data small multiclass dataset
# path_train = "data/selected_512_multiclass/selected_tiles_512_100_20_42/train"
# path_test = "data/selected_512_multiclass/selected_tiles_512_100_20_42/test"

# path to the small mono class large dataset
# path_train = "data/small_large/train"
# path_test = "data/small_large/test"



# calling the model.
model = Unet(
    output_classes=OUTPUT_CLASSES,
    input_shape=INPUT_SHAPE,
    drop_out=True,
    drop_out_rate={"512": 0.3, "256": 0.35, "128": 0.4, "64": 0.45},
    multiclass=bool(OUTPUT_CLASSES - 1),
)

# Load previous weights
# path_checkpoint = "logs/12_16_2021_22_54_01/checkpoint.ckpt"
# model.load_weights(path_checkpoint)


# Starting the logs

log = Logs()

log.main_log(
    comment=COMMENT,
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

SAVE_WEIGHTS_ONLY = True

if OUTPUT_CLASSES > 1:
    MULTICLASS = True
    loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metric_list = [
        "sparse_categorical_accuracy",
    ]
    metrics = [sparse_categorical_accuracy]
    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=log.checkpoint_filepath,
        save_weights_only=SAVE_WEIGHTS_ONLY,
        monitor="val_sparse_categorical_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
else:
    MULTICLASS = False
    loss = tensorflow.keras.losses.BinaryCrossentropy(from_logits=False)
    metric_list = [
        "accuracy",
        "recall",
        "precision",
    ]
    metrics = [binary_accuracy, precision, recall]
    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=log.checkpoint_filepath,
        save_weights_only=SAVE_WEIGHTS_ONLY,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

dl_train = DataLoader(
    PATH_TRAIN,
    batch_size=BATCH_SIZE,
    input_shape=INPUT_SHAPE,
    legacy_mode=False,
    multiclass=MULTICLASS,
)


dl_test = DataLoader(
    PATH_TEST,
    batch_size=BATCH_SIZE,
    input_shape=INPUT_SHAPE,
    legacy_mode=False,
    multiclass=MULTICLASS,
)


dl_train.load(buffer_size=500)
dl_test.load(shuffle=False)

train_batches = dl_train.dataset
test_batches = dl_test.dataset


print("data loaded")


# Preparing the model to be saved using a checkpoint


# Prepare the tensorboard

tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
    log_dir=log.tensorboard_path,
    histogram_freq=1,
    #write_graph=True,
    write_graph=False,
)


# Parameters for early stopping
early_stopping = tensorflow.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
)


print("callbacks defined")

# compiling the model
LEARNING_RATE = 0.001
opt = tensorflow.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


model.compile(optimizer=opt, loss=loss, metrics=metrics)


print("compiling done")
# training the model.

steps_per_epoch = dl_train.n_samples // BATCH_SIZE
validation_steps = max(dl_test.n_samples // BATCH_SIZE, 1)


history = model.fit(
    train_batches,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    validation_data=test_batches,
    callbacks=[
        model_checkpoint_callback,
        tensorboard_callback,
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


log.show_predictions(
    dataset=dl_test.dataset,
    model=model,
    num_batches=NUM_BATCHES,
    multiclass=MULTICLASS,
)

# print("first fitting round")
# tensorflow.keras.backend.clear_session()
# print("Starting fine tuning")

# model_dict = model.get_config()

# model_dict["fine_tune_at"] = 4
# model_dict["upstack_trainable"] = True

#  Second you create the model from the configuration dictionary. This creates a new models with the same layer configuration
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
accuracy = tensorflow.keras.metrics.BinaryAccuracy()
precision = tensorflow.keras.metrics.Precision()
recall = tensorflow.keras.metrics.Recall()

iterator = iter(dl_test.dataset)

for batch_image, batch_target in iterator:
    y_pred_prob = model.predict(batch_image)
    y_pred_multi = np.argmax(y_pred_prob, axis=-1)
    y_pred_multi = y_pred_multi.squeeze()
    y_true_multi = batch_target.numpy()
    y_pred_one = (y_pred_multi > 0).astype(int)
    y_true_one = (y_true_multi > 0).astype(int)
    y_true_one = y_true_one.squeeze()
    accuracy.update_state(y_true_one, y_pred_one)
    precision.update_state(y_true_one, y_pred_one)
    recall.update_state(y_true_one, y_pred_one)

print("For the binary classifier, induced by the categorical classifier:")
print("Binary accuracy:", accuracy.result().numpy())
print("Precision", precision.result().numpy())
print("Recall", recall.result().numpy())


tensorflow.keras.backend.clear_session()
