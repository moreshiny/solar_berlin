# For saving models, refere to
# https://www.tensorflow.org/guide/keras/save_and_serialize
"""This script is to be used to launch the training of the subclassed Unet model.
It defines the parameters of the training. This calls the subclassed model Unet,
the dataloader for producing a tensorflow dataset, and the logging class for logging the experiment.
Args:
OUTPUT_CLASSES, int: Number of classes of the problems. For a binary problems, set to 1.
INPUT_SHAPE, 3-Tuple: Input shape of the input pictures. The shape needs to be divisible by 2^4.
EPOCHS, int: number of training epochs.
BATCH_SIZE, int: Batchsize for the training.
NUM_BATCH, int: Number of batches from which predicted snapshots will be produced by the end
 of the training.
COMMENT, string: Comment on the experiment tested during the training.
PATH_TRAIN, string: path to the train folder containing the input pictures compatile with
the dataloader format.
PATH_TRAIN, string: path to the test folder containing the input pictures compatile with
the dataloader format.
"""
import tensorflow

from roof.dataloader import DataLoader
from unet.unet_resnet101v2 import Unet
from roof.logging import Logs

# Importing the model class
from unet.unet_resnet101v2 import Unet

# Importing the loggind class. The parameters of the training will be saved in a main log,,
# During the training the model is saved in a log subfolder, ath the end of the training,
# snapshots from the test predictions are producted, and the graphs of the metrics of the
# training are dropped in the corresponding local log folder.
from roof.logging import Logs

tensorflow.keras.backend.clear_session()

# parameters of the model.
<<<<<<< HEAD
output_classes = 5  # number of categorical classes.
input_shape = (512, 512, 3)  # input size
epochs = 35

# parameters of the model.
OUTPUT_CLASSES = 5  # number of categorical classes. For 2 classes = 1.
INPUT_SHAPE = (512, 512, 3)  # input size
EPOCHS = 35
PATIENCE = 7

BATCH_SIZE = 8  # batchsize

NUM_BATCHES = 10  # number of batches


COMMENT = "Tested on the clean 8000, first run with erosion/dilation,\n\
    standard learning rate, best dropout.\n\
    Testing as a multiclassifier. First test of,\n\
    erosion and dilation.   "

# Path to data
PATH_TRAIN = "data/bin_clean_8000/train"
PATH_TEST = "data/bin_clean_8000/test"
=======
OUTPUT_CLASSES = 5  # number of categorical classes. for 2 classes = 1.
INPUT_SHAPE = (512, 512, 3)  # input size
EPOCHS = 35

BATCH_SIZE = 8  # batchsize
# Path to the data large multiclass dataset
# path_train = "data/selected_tiles_512_4000_1000_42_partial/train"
# path_test = "data/selected_tiles_512_4000_1000_42_partial/test"

# Path to the data small multiclass dataset
# path_train = "data/selected_512_multiclass/selected_tiles_512_100_20_42/train"
# path_test = "data/selected_512_multiclass/selected_tiles_512_100_20_42/test"
>>>>>>> ff249c3 (updated requirements with pandas)

# path to the small mono class large dataset
# path_train = "data/small_large/train"
# path_test = "data/small_large/test"

# Path to data for Daniel local machine: Half dataset
PATH_TRAIN = "data/cleaned_4000_extract/train"
PATH_TEST = "data/cleaned_4000_extract/test"


# calling the model.
model = Unet(
    output_classes=OUTPUT_CLASSES,
    input_shape=INPUT_SHAPE,
    drop_out=True,
<<<<<<< HEAD
    drop_out_rate={"512": 0.275, "256": 0.3, "128": 0.325, "64": 0.35},
    multiclass=bool(output_classes - 1),
=======
    drop_out_rate={"512": 0.3, "256": 0.35, "128": 0.4, "64": 0.45},
    multiclass=bool(OUTPUT_CLASSES - 1),
>>>>>>> ff249c3 (updated requirements with pandas)
)

# Starting the logs

log = Logs()
<<<<<<< HEAD
=======
COMMENT = "Full large dataset, with the 20pc highst loss cleaned with the \n\
    with the latest model, multiclassification problem ,\n\
    standard learning rate."

>>>>>>> ff249c3 (updated requirements with pandas)
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
mae = tensorflow.keras.losses.MeanSquaredError(name="mae")
recall = tensorflow.keras.metrics.Recall(name="recall")
precision = tensorflow.keras.metrics.Precision(name="precision")

<<<<<<< HEAD
tp = tensorflow.keras.metrics.TruePositives()
fn = tensorflow.keras.metrics.FalseNegatives()
fp = tensorflow.keras.metrics.FalsePositives()


=======
>>>>>>> ff249c3 (updated requirements with pandas)
if OUTPUT_CLASSES > 1:
    MULTICLASS = True
    loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # metrics = [sparse_categorical_accuracy]
    metrics = [mae]
    metric_list = [metric.name for metric in metrics]
    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=log.checkpoint_filepath,
        save_weights_only=False,
        monitor="val_sparse_categorical_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
else:
    MULTICLASS = False
    loss = tensorflow.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [binary_accuracy, precision, recall, tp, fn, fp]
    metric_list = [metric.name for metric in metrics]
    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=log.checkpoint_filepath,
        save_weights_only=False,
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


# Preparing the model to be saved using a checkpoint


# Prepare the tensorboard

tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
    log_dir=log.tensorboard_path,
    histogram_freq=1,
    write_graph=True,
)

PATIENCE = 7
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

<<<<<<< HEAD
num_batches = 3  # number of batches
=======
NUM_BATCHES = 5  # number of batches
>>>>>>> ff249c3 (updated requirements with pandas)
log.show_predictions(
    dataset=dl_test.dataset,
    model=model,
    num_batches=NUM_BATCHES,
    multiclass=MULTICLASS,
)

<<<<<<< HEAD
=======
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

>>>>>>> ff249c3 (updated requirements with pandas)

tensorflow.keras.backend.clear_session()
