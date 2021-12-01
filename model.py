"""Define the UNET model. BAsed in part on the image segmentation notebook
https://www.tensorflow.org/tutorials/images/segmentation
"""
import os
import glob
from datetime import datetime
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from dataloader import DataLoader


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
        model_name: str = "mobilenetv2",
        include_top: bool = True,
        alpha: float = 1,
        pooling: str = None,
        fine_tune_at: int = 0,
        drop_out: bool = False,
        drop_out_rate: float = 0,
    ) -> None:
        """Instantiate the class.
        Args:
            dataset_train (tensorflow.dataset): batched training data
            dataset_train (tensorflow.dataset): batched validation data
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
        self._fine_tune_epochs = fine_tune_epoch

        # save the layer information for the unet model
        self.input_shape = input_shape  # default for RGB images size 224
        self.layers = layer_names  # layers to be used in the up stack
        self._batch_size = batch_size

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
        self._fine_tune_at = fine_tune_at
        self._dropout = drop_out
        self._dropout_rate = drop_out_rate

        # auxiliary variables
        self.model = None
        self._base_model = None
        self._model_history = None
        self._model_history_fine = None
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
        print("compiling")
        self.model = self._compile_model()  # start the model
        print("compiling done")

        # use all train data in batches in each epoch (at least 1 step)
        steps_per_epoch = max(self._n_train // self._batch_size, 1)
        # use all validation data in batches (at least 1 batch)
        validation_steps = max(self._n_val // self._batch_size, 1)

        # write the main log
        self._current_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.logging(comment)
        # prepare model pickeling
        checkpoint_filepath = self._path_log + self._current_time + "/checkpoint"
        model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )
        # Prepare the tensorboard
        log_dir = "logs/tensorboard/" + self._current_time
        tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        # Parameters for early stopping
        early_stopping = (
            tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        )

        # fit the model
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

        self._loss = self._model_history.history["loss"]
        self._val_loss = self._model_history.history["val_loss"]
        self._accuracy = self._model_history.history["accuracy"]
        self._val_accuracy = self._model_history.history["val_accuracy"]

        fine_tune_epochs = self._fine_tune_epochs

        if fine_tune_epochs > 0:

            self.freezing_layers()

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
            self._loss += self._model_history_fine.history["loss"]
            self._val_loss += self._model_history_fine.history["val_loss"]
            self._accuracy += self._model_history_fine.history["accuracy"]
            self._val_accuracy += self._model_history_fine.history["val_accuracy"]

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

    def _compile_model(self) -> tensorflow.keras.Model:
        """
        Takes the Unet models, and compile the model. Return the model.

        Returns:
            the compiled model, with the ADAM gradient descent, binary
            crossentropy loss, and accuracy metrics.
        """
        print("Model built")
        model = self._setup_unet_model(
            output_channels=self.output_classes,
        )

        model.compile(
            optimizer="adam",
            loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def _setup_unet_model(
        self,
        output_channels: int,
    ) -> tensorflow.keras.Model:
        """
        Unet model from the segmentation notebook by tensorflow. Define the model.

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
        inputs = tensorflow.keras.layers.Input(tensorflow.TensorShape(self.input_shape))

        # downsampling through the model
        # needs to have base model defined.
        down_stack = tensorflow.keras.Model(
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
                drop_out_rate=self._dropout_rate,
            ),  # 4x4 -> 8x8
            self._upsample(
                256,
                3,
                apply_dropout=drop_out,
                drop_out_rate=self._dropout_rate,
            ),  # 8x8 -> 16x16
            self._upsample(
                128,
                3,
                apply_dropout=drop_out,
                drop_out_rate=self._dropout_rate,
            ),  # 16x16 -> 32x32
            self._upsample(
                64,
                3,
                apply_dropout=drop_out,
                drop_out_rate=self._dropout_rate,
            ),  # 32x32 -> 64x64
        ]

        for up, skip in zip(up_stack, skips):
            layer = up(layer)
            concat = tensorflow.keras.layers.Concatenate()
            layer = concat([layer, skip])

        # this is the last layer of the model
        last = tensorflow.keras.layers.Conv2DTranspose(
            filters=output_channels,
            kernel_size=3,
            strides=2,
            padding="same",
        )  # 64x64 -> 128x128

        layer = last(layer)

        print("model built")

        return tensorflow.keras.Model(inputs=inputs, outputs=layer)

    def freezing_layers(self) -> None:
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

        for i, layer in enumerate(self._base_model.layers[-fine_tune_at:-9]):
            layer.trainable = True
            layer_log.write("\n")
            layer_log.write(f"{i} : {layer.name} : trainable = true")

        for i, layer in enumerate(self.model.layers[-9:]):
            layer.trainable = False
            layer_log.write("\n")
            layer_log.write(f"{i} : {layer.name} : trainable = true")

        layer_log.close()

    def _get_base_model(self) -> tensorflow.keras.Model:
        """
        Define the base of the model, MobileNetV2. Note that a discussion on
        the shape is necessary: if the shape of the pictures is the default
        shape (224, 224, 3), the include top options needs to be set to True,
        and the input shape is not passed as an argument of the base model.
        Otherwise the default input shape is used.
        Returns:
            the base model MobileNetV2
        """

        if self._include_top:
            base_model = tensorflow.keras.applications.resnet_v2.ResNet101V2(
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
            )
            self.input_shape = (224, 224, 3)

        else:
            base_model = tensorflow.keras.applications.resnet_v2.ResNet101V2(
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
        initializer = tensorflow.random_normal_initializer(0.0, 0.02)
        result = tensorflow.keras.Sequential()
        result.add(
            tensorflow.keras.layers.Conv2DTranspose(
                filters,
                size,
                strides=2,
                padding="same",
                kernel_initializer=initializer,
                use_bias=False,
            )
        )

        result.add(tensorflow.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tensorflow.keras.layers.Dropout(drop_out_rate))

        result.add(tensorflow.keras.layers.ReLU())

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
            plt.imshow(tensorflow.keras.utils.array_to_img(display_list[i]))
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
        self, dataset: tensorflow.data.Dataset = None, num_batches: int = 1
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
            local_log.write(f"alpha: {self._alpha}")
            local_log.write("\n")
            local_log.write(f"pooling: {self._pooling}")
            local_log.write("\n")
            local_log.write(f"pooling:{self._fine_tune_at}")
            local_log.write("\n")
            local_log.write(f"Dropout: {self._dropout}")
            local_log.write("\n")
            local_log.write(f"Dropout rate: {self._dropout_rate}")
            local_log.write("\n")
            local_log.write(f"Epochs: {self.epochs}")
            local_log.write("\n")
            local_log.write(f"fine tune Epochs: {self.fine_tune_epoch}")
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
            local_log.write(f"Losses:{self._loss}")
            local_log.write("\n")
            local_log.write(f"Val losses:{self._val_loss}")
            local_log.write("\n")

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(self._accuracy, label="Training Accuracy")
            plt.plot(self._val_accuracy, label="Validation Accuracy")
            plt.ylim([0.8, 1])
            plt.plot(
                [self.epochs + 0.5, self.epochs + 0.5],
                plt.ylim(),
                label="Start Fine Tuning",
            )
            plt.legend(loc="lower right")
            plt.title("Training and Validation Accuracy")

            plt.subplot(2, 1, 2)
            plt.plot(self._loss, label="Training Loss")
            plt.plot(self._val_loss, label="Validation Loss")
            plt.ylim([0, 1.0])
            plt.plot(
                [self.epochs + 0.5, self.epochs + 0.5],
                plt.ylim(),
                label="Start Fine Tuning",
            )
            plt.legend(loc="upper right")
            plt.title("Training and Validation Loss")
            plt.xlabel("epoch")

            path_graph = self._path_log + self._current_time + "/losses.pdf"
            plt.savefig(path_graph)
            plt.close()

        dot_img_file = self._path_log + self._current_time + "/model.pdf"

        tensorflow.keras.utils.plot_model(
            self.model,
            to_file=dot_img_file,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=96,
        )

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
