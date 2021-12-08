"""Define the UNET model. BAsed in part on the image segmentation notebook
https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""
import os
import glob
from datetime import datetime
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
<<<<<<< HEAD:unet/unet_basic_nt.py
from roof.dataloader import DataLoader
=======
from loading.dataloader import DataLoader
>>>>>>> b8e175c (Rename subfolders):models/unet_basic_nt.py


class Model:
    """
    The purpose of the class is to define the Unet model,
    and contain the necessary functions (compile, and fit) to run it and save
    the results.
    """

    def __init__(
        self,
        path_train: str,
        path_test: str,
        layer_names: List[str],
        output_classes: int = 1,
        input_shape: Tuple[int] = (224, 224, 3),
        epochs: int = 10,
        batch_size: int = 32,
        model_name: str = "UNET",
        include_top: bool = True,
        alpha: float = 1,
        pooling: str = None,
    ) -> None:
        """Instantiate the class.
        Args:
            dataset_train (tf.dataset): batched training data
            dataset_train (tf.dataset): batched validation data
            layer_names (list of strings): defining the MobilenetV2 NN.
            output_classes (int): number of classes for the classification.
                Defaults to 1.
            input_shape (3-tuple): a 3 tuple defining the shape of the input
                image. Defaults to (224, 224, 3)
            epochs (int): number of epochs to train the model. Defaults to 10.
            batch_size (int): size of the batches in training. Defaults to 32.
        """

        # save the unet model parameters
        self.output_classes = output_classes  # 1 class for binary classification
        self.epochs = epochs

        # save the layer information for the unet model
        self.input_shape = input_shape  # default for RGB images size 224
        self.layers = layer_names  # layers to be used in the up stack
        self._batch_size = batch_size

        # save the training and validation dataloaders
        dl_train = DataLoader(path_train, batch_size=self._batch_size)
        dl_val = DataLoader(path_test, batch_size=self._batch_size)
        dl_train.load()
        dl_val.load()
        self.train_batches = dl_train.dataset
        self.test_batches = dl_val.dataset

        # save size of train and validation data for internal use
        self._n_train = dl_train.n_samples
        self._n_val = dl_val.n_samples

        # paths for logging
        self._path_log = "logs/"
        self._path_main_log_file = self._path_log + "main_log.log"
        self._path_aux = self._path_log + "log.aux"
        self._current_time = ""

        # parameters of the model
        self._model_name = model_name
        self._alpha = alpha
        self._include_top = include_top
        self._pooling = pooling

        # auxiliary variables
        self.model = None
        self._model_history = None
        self._accuracy = []
        self._val_accuracy = []
        self._loss = []
        self._val_loss = []
        self._dictionary_performance = {}

    # TODO what type is this returning?
    def model_history(self, comment: str = "Model running"):
        """
        Train the model. Return the history of the model.

        Args:
            comment (str): take a string, containing the necessary comment on
            the model. Default set to "Model running".

        Returns:
            The fitted model history.
        """
        self.model = self._compile_model()  # start the model

        # use all train data in batches in each epoch (at least 1 step)
        steps_per_epoch = max(self._n_train // self._batch_size, 1)
        # use all validation data in batches (at least 1 batch)
        validation_steps = max(self._n_val // self._batch_size, 1)

        # write the main log
        self._current_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.logging(comment)
        # prepare model pickeling
        checkpoint_filepath = self._path_log + self._current_time + "/checkpoint"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )
        # Prepare the tensorboard
        log_dir = "logs/tensorboard/" + self._current_time
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        # Clearing the output
        tf.keras.backend.clear_session

        # fit the model
        self._model_history = self.model.fit(
            self.train_batches,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.test_batches,
            callbacks=[model_checkpoint_callback, tensorboard_callback],
        )
        self._loss = self._model_history.history["loss"]
        self._val_loss = self._model_history.history["val_loss"]
        self._accuracy = self._model_history.history["accuracy"]
        self._val_accuracy = self._model_history.history["val_accuracy"]

        # log performance
        self._local_log(comment)
        self.saving_model_performance()
        self.show_predictions()

        # delete the model if the val accuracy is worse than existing model.
        max_perf = max(self._dictionary_performance.values())
        if max_perf > max(self._val_accuracy):
            for filename in glob.glob(checkpoint_filepath + "*"):
                os.remove(filename)
        else:
            self.logging_saved_model()

        return self._model_history

    def _compile_model(self) -> tf.keras.Model:
        """
        Takes the Unet models, and compile the model. Return the model.

        Returns:
            the compiled model, with the ADAM gradient descent, binary
            crossentropy loss, and accuracy metrics.
        """
        model = self._setup_unet_model(output_channels=self.output_classes)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        return model

    def _setup_unet_model(self, output_channels: int) -> tf.keras.Model:
        """
        Unet model from the segmentation notebook by tf. Define the model.

        Args:
        output_channels (int): number of categories in the classification.

        Returns:
            The model unet.

        """
        inputs = keras.Input(shape=self.input_shape)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        intermediary = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        intermediary = layers.BatchNormalization()(intermediary)
        intermediary = layers.Activation("relu")(intermediary)

        previous_block_activation = intermediary  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            intermediary = layers.Activation("relu")(intermediary)
            intermediary = layers.SeparableConv2D(filters, 3, padding="same")(
                intermediary
            )
            intermediary = layers.BatchNormalization()(intermediary)

            intermediary = layers.Activation("relu")(intermediary)
            intermediary = layers.SeparableConv2D(filters, 3, padding="same")(
                intermediary
            )
            intermediary = layers.BatchNormalization()(intermediary)

            intermediary = layers.MaxPooling2D(3, strides=2, padding="same")(
                intermediary
            )

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            intermediary = layers.add([intermediary, residual])  # Add back residual
            previous_block_activation = intermediary  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            intermediary = layers.Activation("relu")(intermediary)
            intermediary = layers.Conv2DTranspose(filters, 3, padding="same")(
                intermediary
            )
            intermediary = layers.BatchNormalization()(intermediary)

            intermediary = layers.Activation("relu")(intermediary)
            intermediary = layers.Conv2DTranspose(filters, 3, padding="same")(
                intermediary
            )
            intermediary = layers.BatchNormalization()(intermediary)

            intermediary = layers.UpSampling2D(2)(intermediary)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            intermediary = layers.add([intermediary, residual])  # Add back residual
            previous_block_activation = intermediary  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(
            output_channels, 3, activation="softmax", padding="same"
        )(intermediary)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

    # TODO what type should the elements in the display_list be?
    def _display(self, display_list: List) -> None:
        """Display side by side the pictures contained in the list.

        Args:
            display_list (List): Three images, aerial pictures, true mask,
                                and predicted mask in that order.

        """
        plt.figure(figsize=(15, 15))
        title = ["Input Image", "True Mask", "Predicted Mask"]
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            type(display_list[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis("off")
        path_snapshot = self._path_log + self._current_time + "/snapshots"
        if not os.path.exists(path_snapshot):
            os.mkdir(path_snapshot)
        path_fig = path_snapshot + f"/output{np.random.rand()}.jpeg"
        plt.savefig(path_fig)
        plt.close()

    # TODO what type does this take and return?
    def _create_mask(self, pred_mask):
        """Create a mask from the predicted array.

        Args:
            a predicted image, through the predict method.

        Returns:
            a mask.

        """
        pred_mask = (pred_mask > 0.5).astype(int) * 255
        return pred_mask

    def show_predictions(
        self, dataset: tf.data.Dataset = None, num_batches: int = 1
    ) -> None:
        """Display side by side an earial photography, its true mask, and the
            predicted mask.

        Args:
            A dataset in the form provided by the dataloader.
            num_batches (int): number of batches to display.

        """
        # default to the test dataset
        if dataset is None:
            dataset = self.test_batches

        for image_batch, mask_batch in dataset.take(num_batches):
            pred_mask_batch = self.model.predict(image_batch)
            for image, mask, pred_mask in zip(image_batch, mask_batch, pred_mask_batch):
                self._display([image, mask, self._create_mask(pred_mask)])

    def logging(self, comment: str) -> None:
        """
        Log the model in the main log.

        Args:
            comment (str): the necessary comment on the model.
        """

        if not os.path.exists(self._path_log):
            os.mkdir(self._path_log)

        with open(self._path_main_log_file, "a", encoding="utf-8") as main_log:
            main_log.write("\n")
            main_log.write("------")
            main_log.write("\n")
            main_log.write(self._current_time)
            main_log.write("\n")
            main_log.write(comment)
            main_log.write("\n")
            main_log.write("\n")
            main_log.close()

    def _local_log(self, comment: str) -> None:
        """
        Create the local log.

        Args:
            comment (str): the necessary comment on the model.
        """

        if not os.path.exists(self._path_log + self._current_time):
            os.mkdir(self._path_log)

        path_local_log = self._path_log + self._current_time + "/local_log.log"

        with open(path_local_log, "a", encoding="utf-8") as local_log:
            local_log.write(self._current_time)
            local_log.write("\n")
            local_log.write(comment)
            local_log.write("\n")
            local_log.write(f"Model name: {self._model_name}")
            local_log.write("\n")
            local_log.write(f"include top: {self._include_top}")
            local_log.write("\n")
            local_log.write(f"Alpha: {self._alpha}")
            local_log.write("\n")
            local_log.write(f"Image size: {self.input_shape}")
            local_log.write("\n")
            local_log.write(f"Train size: {self._n_train}")
            local_log.write("\n")
            local_log.write(f"Validation size: {self._n_val}")
            local_log.write("\n")
            local_log.write(f"Epochs:{self.epochs}")
            local_log.write("\n")
            local_log.write(f"Batches:{self._batch_size}")
            local_log.write("\n")
            local_log.write(f"Struture of the network: {self.layers}")
            local_log.write("\n")
            local_log.write(f"Accuracy:{self._accuracy}")
            local_log.write("\n")
            local_log.write(f"Val Accuracy:{self._val_accuracy}")
            local_log.write("\n")
            local_log.write(f"Losses:{self._loss}")
            local_log.write("\n")
            local_log.write(f"Val losses:{self._val_loss}")
            local_log.write("\n")

        plt.plot(
            self._model_history.epoch,
            self._accuracy,
            "r",
            label="Training accuracy",
        )
        plt.plot(
            self._model_history.epoch,
            self._val_accuracy,
            "bo",
            label="Validation accuracy",
        )
        plt.title("Training and Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("accuracy")
        plt.ylim([0, 1])
        plt.legend()
        path_graph = self._path_log + self._current_time + "/accuracy.pdf"
        plt.savefig(path_graph)
        plt.close()
        plt.plot(
            self._model_history.epoch,
            self._loss,
            "r",
            label="Training loss",
        )
        plt.plot(
            self._model_history.epoch,
            self._val_loss,
            "bo",
            label="Validation loss",
        )
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.ylim([0, 1])
        plt.legend()
        path_graph = self._path_log + self._current_time + "/losses.pdf"
        plt.savefig(path_graph)
        plt.close()

    def saving_model_performance(self) -> None:
        """save the performance (accuracy) of the model. Print these
        performance in an auxiliary files.
        """

        with open(self._path_aux, "a", encoding="utf-8") as aux_file:
            aux_file.write(f"{self._current_time} : {max(self._val_accuracy)}")
            aux_file.write("\n")

        with open(self._path_aux, "r", encoding="utf-8") as aux_file:
            for line in aux_file:
                (key, val) = line.split(":")
                self._dictionary_performance[key] = float(val)

    def logging_saved_model(self) -> None:
        """Log in the main file tha the model has been saved."""

        with open(self._path_main_log_file, "a", encoding="utf-8") as main_log:
            main_log.write("Model saved!")
            main_log.write("\n")
            main_log.write(f"Validation accuracy: {max(self._val_accuracy)}")
