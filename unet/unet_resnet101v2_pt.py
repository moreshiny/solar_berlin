"""Define the UNET model. BAsed in part on the image segmentation notebook
https://www.tf.org/tutorials/images/segmentation
"""
import os
import glob
import shutil
from datetime import datetime
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

<<<<<<< HEAD:unet/unet_resnet101v2_pt.py
<<<<<<< HEAD:unet/unet_resnet101v2_pt.py
from roof.dataloader import DataLoader
=======
from loading.dataloader import DataLoader
>>>>>>> b8e175c (Rename subfolders):models/unet_resnet101v2_pt.py
=======
from roof.dataloader import DataLoader
>>>>>>> bdb2ba5 (Combine classes into a single roof module):models/unet_resnet101v2_pt.py


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
        fine_tune_epoch: int = 10,
        batch_size: int = 32,
        model_name: str = "Unet",
        include_top: bool = True,
        pooling: str = None,
        fine_tune_at: int = 0,
        drop_out: bool = False,
        drop_out_rate: dict = {"512": 0, "256": 0, "128": 0, "64": 0},
        patience: int = 10,
        patience_fine_tune: int = 10,
        buffer_size: int = 500,
    ) -> None:
        """Instantiate the class.
        Args:
            path_train: str: path to the train dataset folder
            path_test: str: path to the test dataset folder
            layer_names: List[str]: list of layers doing the skip connections in the unet
            output_classes: int = 1: number of categories in the classification
            input_shape: Tuple[int] = (224, 224, 3): input size of the image, defaukt to (224, 224, 3).
            epochs: int = 10: number of epochs training before finie tuning, default to 10.
            fine_tune_epoch: int = 10: number of epochs in the fine tuning epochs, default to 10
            batch_size: int = 32: batche size. Default to 32.
            model_name: str = "Unet"; Model name, Default to unet.
            include_top: bool = True. Wether to include the claissification layer of the pretrained network.\
                Default to False.
            pooling: str = None,: Pooling in the convolution layers, default to None.
            fine_tune_at: int = 0, Number of layers at the top of the network to be fine tuned. Default to 0.
            drop_out: bool = False. Activating the dropout in the up-sample stacks; DAfault to False.
            drop_out_rate: dict. Dropout rate in the up-sample stack, in the form of a dictionary with keys\
                512, 256, 128, 64. Default to {"512": 0, "256": 0, "128": 0, "64": 0}.
            patience: int = 10. Time before early stopping of the training. Default to ten.
            patience_fine_tuning: int = 10. Time before early stopping of the training after fine tuning.\
                Default to ten.
            bufer_size: int = 500. Size of the shuffle buffer. Defaults to 500.
        """

        # save the unet model parameters
        self.output_classes = output_classes  # 1 class for binary classification
        self.epochs = epochs
        self._fine_tune_epochs = fine_tune_epoch
        self._trained_base_epochs = 0
        self._trained_including_fine_tune = 0

        # save the layer information for the unet model
        self.input_shape = input_shape  # default for RGB images size 224
        self.layers = layer_names  # layers to be used in the up stack
        self._batch_size = batch_size
        self._buffer_size = buffer_size

        # save the training and validation dataloaders
        dl_train = DataLoader(
            path_train,
            batch_size=self._batch_size,
            input_shape=self.input_shape,
        )
        dl_val = DataLoader(
            path_test,
            batch_size=self._batch_size,
            input_shape=self.input_shape,
        )
        dl_train.load(buffer_size=self._buffer_size)
        dl_val.load(buffer_size=self._buffer_size)
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
        self._include_top = include_top
        self._pooling = pooling
        self._fine_tune_at = fine_tune_at
        self._dropout = drop_out
        self._dropout_rate = drop_out_rate
        self._patience = patience
        self._patience_fine_tune = patience_fine_tune

        # auxiliary variables
        self.model = None
        self._base_model = None
        self._model_history = None
        self._model_history_fine = None
        self._accuracy = []
        self._val_accuracy = []
        self._loss = []
        self._val_loss = []
        self._precision = []
        self._val_precision = []
        self._recal = []
        self._val_recall = []
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
        print("compiling")
        # Calling the model.
        self.model = self._setup_unet_model(
            output_channels=self.output_classes,
        )
        print("Model called")

        # Compile the model.
        self._compile_model(self.model)  # start the model
        print("first compiling done")

        # use all train data in batches in each epoch (at least 1 step)
        steps_per_epoch = max(self._n_train // self._batch_size, 1)
        # use all validation data in batches (at least 1 batch)
        validation_steps = max(self._n_val // self._batch_size, 1)

        # write the main log
        self._current_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self._logging(comment)
        # prepare model pickeling
        checkpoint_filepath = (
            self._path_log + self._current_time + "/model/checkpoint.ckpt"
        )

        # checkpoint_dir = os.path.dirname(checkpoint_filepath)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )

        # Prepare the tensorboard
        log_dir = "logs/tensorboard/" + self._current_time
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        # Parameters for early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self._patience,
        )

        # fit the model
        print("Training the model")
        self._model_history = self.model.fit(
            self.train_batches,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.test_batches,
            callbacks=[
                model_checkpoint_callback,
                tensorboard_callback,
                early_stopping,
            ],
        )

        # Saving the basic data of the model.
        self._loss = self._model_history.history["loss"]
        self._val_loss = self._model_history.history["val_loss"]
        self._accuracy = self._model_history.history["accuracy"]
        self._val_accuracy = self._model_history.history["val_accuracy"]
        self._precision = self._model_history.history["precision"]
        self._val_precision = self._model_history.history["val_precision"]
        self._recall = self._model_history.history["recall"]
        self._val_recall = self._model_history.history["val_recall"]
        self._trained_base_epochs = len(self._loss)

        fine_tune_epochs = self._fine_tune_epochs

        # DEmarring the fine tuning.
        if fine_tune_epochs > 0:
            print("Fine tuning training starting")

            # Freezing the layers
            self._freezing_layers()

            # Loading the last best state of the model. Loading the weight only to prevent overwriting the trainability of the layers
            self.model.load_weights(checkpoint_filepath)

            print("Compiling the model in fine tuning mode")

            #  Prepare the compilation of the model in the fine tuning training.
            self._compile_model(
                model=self.model,
                fine_tune_epochs=fine_tune_epochs,
                learning_rate=0.0001,
            )
            print("Training the model in fine tuning mode")

            # Defining the early stopping for the model in the fine tuning epochs.
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self._patience_fine_tune,
            )

            # Training the model in the fine tuning epochs.
            self._model_history_fine = self.model.fit(
                self.train_batches,
                epochs=self.epochs + self._fine_tune_epochs,
                initial_epoch=self._model_history.epoch[-1],
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                validation_data=self.test_batches,
                callbacks=[
                    model_checkpoint_callback,
                    tensorboard_callback,
                    early_stopping,
                ],
            )
            # Saving the parameters of the model
            self._loss += self._model_history_fine.history["loss"]
            self._val_loss += self._model_history_fine.history["val_loss"]
            self._accuracy += self._model_history_fine.history["accuracy"]
            self._val_accuracy += self._model_history_fine.history["val_accuracy"]
            self._precision += self._model_history_fine.history["precision"]
            self._val_precision += self._model_history_fine.history["val_precision"]
            self._recall += self._model_history_fine.history["recall"]
            self._val_recall += self._model_history_fine.history["val_recall"]
            self._trained_including_fine_tune = len(self._loss)

        # log performance
        self._local_log(comment)
        self.saving_model_performance()
        self.show_predictions()

        # Save the model if the val accuracy is as good as the one of an existing model.
        # max_perf = max(self._dictionary_performance.values())
        # if max_perf <= max(self._val_accuracy):
        self._logging_saved_model()
        # self._make_archive()

        # Erase the temporary folder where the model is saved
        # shutil.rmtree(self._path_log + self._current_time + "/model")

        return self._model_history

    def _compile_model(
        self,
        model: tf.keras.Model,
        fine_tune_epochs: int = 0,
        learning_rate: float = 0.001,
    ) -> tf.keras.Model:
        """
        Takes the Unet models, and compile the model. Return the model.

        Returns:
            the compiled model, with the ADAM gradient descent, binary
            crossentropy loss, and accuracy metrics.
        """
        print("Model compiled")

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                "accuracy",
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.Precision(name="precision"),
            ],
        )

        return model

    def _setup_unet_model(
        self,
        output_channels: int,
    ) -> tf.keras.Model:
        """
        Unet model from the segmentation notebook by tf. Define the model.

        Args:
        output_channels (int): number of categories in the classification.

        Returns:
            The model unet.

        """
        print("building model")
        # initiate the base model
        self._base_model = self._get_base_model()

        base_model = self._base_model

        base_model.trainable = False

        # select the requested down stack layers
        selected_output_layers = [
            base_model.get_layer(name).output for name in self.layers
        ]

        # define the input layer
        inputs = tf.keras.layers.Input(tf.TensorShape(self.input_shape))

        # downsampling through the model
        # needs to have base model defined.
        down_stack = tf.keras.Model(
            inputs=base_model.input,
            outputs=selected_output_layers,
        )
        # freeze the downstack layers
        down_stack.trainable = False

        skips = down_stack(inputs)
        layer = skips[-1]
        skips = reversed(skips[:-1])

        # upsampling and establishing the skip connections
        drop_out = self._dropout
        up_stack = [
            self._upsample(
                512,
                3,
                apply_dropout=drop_out,
                drop_out_rate=self._dropout_rate["512"],
            ),  # 4x4 -> 8x8
            self._upsample(
                256,
                3,
                apply_dropout=drop_out,
                drop_out_rate=self._dropout_rate["256"],
            ),  # 8x8 -> 16x16
            self._upsample(
                128,
                3,
                apply_dropout=drop_out,
                drop_out_rate=self._dropout_rate["128"],
            ),  # 16x16 -> 32x32
            self._upsample(
                64,
                3,
                apply_dropout=drop_out,
                drop_out_rate=self._dropout_rate["64"],
            ),  # 32x32 -> 64x64
        ]

        for up, skip in zip(up_stack, skips):
            layer = up(layer)
            concat = tf.keras.layers.Concatenate()
            layer = concat([layer, skip])

        # this is the last layer of the model
        last_conv = tf.keras.layers.Conv2DTranspose(
            filters=output_channels,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="sigmoid",
        )  # 64x64 -> 128x128

        layer = last_conv(layer)

        # resizing after the dilation
        # TODO Hardcoded size!!!
        # layer = tf.keras.layers.Resizing(
        #    1024,
        #    1024,
        #    interpolation="bilinear",
        #    crop_to_aspect_ratio=False,
        # )(layer)

        # Implementing dilation
        # filters = tf.constant(
        #    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        # )
        # filters = tf.expand_dims(filters, axis=-1)
        # rate_height = 1
        # rate_width = 1
        # stride_height = 1
        # stride_width = 1

        # layer = tf.nn.dilation2d(
        #    input=layer,
        #    filters=filters,
        #    strides=[1, stride_height, stride_width, 1],
        #    padding="SAME",
        #    data_format="NHWC",
        #    dilations=[1, rate_height, rate_width, 1],
        # )

        print("model built")

        return tf.keras.Model(inputs=inputs, outputs=layer)

    def _freezing_layers(self) -> None:
        """
        Freezing the last layers of the model.

        """
        path_layer_log = self._path_log + self._current_time + "/layer_log.log"
        layer_log = open(path_layer_log, "a", encoding="utf-8")

        fine_tune_at = self._fine_tune_at

        self._base_model.trainable = True

        for i, layer in enumerate(self._base_model.layers[0:-fine_tune_at]):
            layer.trainable = False
            layer_log.write("\n")
            layer_log.write(f"{i} : {layer.name} : trainable = false")

        for i, layer in enumerate(self._base_model.layers[-fine_tune_at:]):
            layer.trainable = True
            layer_log.write("\n")
            layer_log.write(f"{i} : {layer.name} : trainable = true")

        for i, layer in enumerate(self.model.layers[-9:]):
            layer.trainable = False
            layer_log.write("\n")
            layer_log.write(f"{i} : {layer.name} : trainable = false")

        layer_log.close()

    def _get_base_model(self) -> tf.keras.Model:
        """
        Define the base of the model, Resnetv2_101. Note that a discussion on
        the shape is necessary: if the shape of the pictures is the default
        shape (224, 224, 3), the include top options needs to be set to True,
        and the input shape is not passed as an argument of the base model.
        Otherwise the default input shape is used.
        Returns:
            the base model Resnetv2_101
        """

        if self._include_top:
            base_model = tf.keras.applications.resnet_v2.ResNet101V2(
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
            )
            self.input_shape = (224, 224, 3)

        else:
            base_model = tf.keras.applications.resnet_v2.ResNet101V2(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=self.input_shape,
                pooling=max,
            )

        return base_model

    # TODO what types are filters size (int?) and what type does this return?
    def _upsample(
        self,
        filters,
        size,
        apply_dropout: bool = False,
        drop_out_rate: float() = 0,
    ):
        """Define the upsampling stack. Introduced as an alternative to the
            pix2pix implementation of the tensoflow notebook.
           Credit: https://www.tensorflow.org/tutorials/generative/pix2pix
           Conv2DTranspose => Dropout => Relu
        Args:
            filters: number of filters
            size: filter size
            apply_dropout: If True, adds the dropout layer
        Returns:
               Upsample Sequential Model
        """
        initializer = tf.random_normal_initializer(0.0, 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(
                filters,
                size,
                strides=2,
                padding="same",
                kernel_initializer=initializer,
                use_bias=False,
            )
        )

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(drop_out_rate))

        result.add(tf.keras.layers.ReLU())

        return result

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

    def _logging(self, comment: str) -> None:
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
            local_log.write(f"Image size: {self.input_shape}")
            local_log.write("\n")
            local_log.write(f"Train size: {self._n_train}")
            local_log.write("\n")
            local_log.write(f"Validation size: {self._n_val}")
            local_log.write("\n")
            local_log.write(f"pooling: {self._pooling}")
            local_log.write("\n")
            local_log.write(f"Unfrozen layer: {self._fine_tune_at}")
            local_log.write("\n")
            local_log.write(f"Dropout: {self._dropout}")
            local_log.write("\n")
            local_log.write(f"Dropout rate: {self._dropout_rate}")
            local_log.write("\n")
            local_log.write(f"Programmed base epochs: {self.epochs} ")
            local_log.write(
                f" trained base epochs: {self._trained_including_fine_tune}",
            )
            local_log.write("\n")
            local_log.write(f"Programmed fine tune epochs: {self._fine_tune_epochs}  ")
            local_log.write(
                f"trained fine tune epochs = {self._trained_including_fine_tune}"
            )
            local_log.write("\n")
            local_log.write(f"Batches:{self._batch_size}")
            local_log.write("\n")
            local_log.write(f"Struture of the network: {self.layers}")
            local_log.write("\n")
            local_log.write(f"fine tuned layer: {self._fine_tune_at}")
            local_log.write("\n")
            local_log.write(f"Accuracy:{self._accuracy}")
            local_log.write("\n")
            local_log.write(f"Val Accuracy:{self._val_accuracy}")
            local_log.write("\n")
            local_log.write(f"Precision:{self._precision}")
            local_log.write("\n")
            local_log.write(f"Val Precision:{self._val_precision}")
            local_log.write("\n")
            local_log.write(f"Recall:{self._recall}")
            local_log.write("\n")
            local_log.write(f"Val Recall:{self._val_recall}")
            local_log.write("\n")
            local_log.write(f"Losses:{self._loss}")
            local_log.write("\n")
            local_log.write(f"Val losses:{self._val_loss}")
            local_log.write("\n")

            plt.figure(figsize=(8, 16))
            plt.subplot(4, 1, 1)
            plt.plot(self._accuracy, label="Training Accuracy")
            plt.plot(self._val_accuracy, label="Validation Accuracy")
            plt.ylim([0.8, 1])
            plt.plot(
                [self._trained_base_epochs + 0.5, self._trained_base_epochs + 0.5],
                plt.ylim(),
                label="Start Fine Tuning",
            )
            plt.legend(loc="lower right")
            plt.title("Training and Validation Accuracy")

            plt.subplot(4, 1, 2)
            plt.plot(self._precision, label="Precision")
            plt.plot(self._val_precision, label="Validation Precision")
            plt.ylim([0, 1.0])
            plt.plot(
                [self._trained_base_epochs + 0.5, self._trained_base_epochs + 0.5],
                plt.ylim(),
                label="Start Fine Tuning",
            )
            plt.legend(loc="upper right")
            plt.title("Training and Validation Precision")

            plt.subplot(4, 1, 3)
            plt.plot(self._recall, label="Recall")
            plt.plot(self._val_recall, label="Validation Recall")
            plt.ylim([0, 1.0])
            plt.plot(
                [self._trained_base_epochs + 0.5, self._trained_base_epochs + 0.5],
                plt.ylim(),
                label="Start Fine Tuning",
            )
            plt.legend(loc="lower right")
            plt.title("Training and Validation Recall")

            plt.subplot(4, 1, 4)
            plt.plot(self._loss, label="Training Loss")
            plt.plot(self._val_loss, label="Validation Loss")
            plt.ylim([0, 1.0])
            plt.plot(
                [self._trained_base_epochs + 0.5, self._trained_base_epochs + 0.5],
                plt.ylim(),
                label="Start Fine Tuning",
            )
            plt.legend(loc="upper right")
            plt.title("Training and Validation Loss")
            plt.xlabel("epoch")

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

    def _logging_saved_model(self) -> None:
        """Log in the main file tha the model has been saved."""

        with open(self._path_main_log_file, "a", encoding="utf-8") as main_log:
            main_log.write("Model saved!")
            main_log.write("\n")
            main_log.write(f"Validation accuracy: {max(self._val_accuracy)}")

    def _make_archive(self):
        if os.path.exists(self._path_log + self._current_time + "/model"):
            print("Compressing model")
            shutil.make_archive(
                self._path_log + self._current_time + "/model",
                "zip",
                self._path_log + self._current_time,
                "model",
            )
