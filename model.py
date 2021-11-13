"""Define the UNET model. BAsed in part on the image segmentation notebook
https://www.tensorflow.org/tutorials/images/segmentation
"""
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.losses import BinaryCrossentropy
from dataloader import DataLoader

# from IPython.display import clear_output use in the display call back.


class Model:
    """
    The purpose of the class is to define the Unet model,
    and contain the necessary functions (compile, and fit) to run it.
    """

    def __init__(
        self,
        path_train,
        path_test,
        layer_names,
        output_classes=1,
        input_shape=(224, 224, 3),
        epochs=10,
    ) -> None:
        """Instantiate the class.
        Args:
            dataset_train (tf.dataset): batched training data
            dataset_train (tf.dataset): batched validation data
            layer_names (list of strings): list of strings defining the MobilenetV2 NN.
            output_classes (int): interget defining the number of classes for the
            classification. Default to 2.
            input_shape (3-tuple): a 3 tuple defining the shape of the input image.
            Default set to (224, 224, 3)
        """
        # Initialisation missing.
        # save the unet model parameters
        self.output_classes = output_classes  # 2 classes for binary classification
        self.epochs = epochs
        # save the layer information for the unet model
        # TODO: can we infer include_top from input_shape? Yes. I have modified this.
        # default True, use the default input layer
        self.input_shape = input_shape  # default for RGB images size 224
        self.layers = layer_names  # layers to be used in the up stack
        ###
        self._batch_size = 32
        dl_train = DataLoader(path_train, batch_size=self._batch_size)
        dl_val = DataLoader(path_test, batch_size=self._batch_size)
        dl_train.load()
        dl_val.load()
        self.train_batches = dl_train.dataset
        self.test_batches = dl_val.dataset
        # save size of train and validation data for internal use
        self._n_train = dl_train.dataset_input.cardinality().numpy()
        self._n_val = dl_val.dataset_target.cardinality().numpy()
        # self._batch_size = dl_train.batch_size
        # paths for logging
        self._path_log = "logs/"
        self._path_main_log_file = self._path_log + "main_log.log"
        self._path_aux = self._path_log + "log.aux"
        self._current_time = ""
        # Auxiliary variables
        self.model = None
        self._model_history = None
        self._accuracy = []
        self._val_accuracy = []
        self._loss = []
        self._val_loss = []
        self._dictionary_performance = {}

    def model_history(self, comment="Model running"):
        """
        Train the model. Return the history of the model.

        Args:
            comment (str): take a string, containing the necessary comment on the model.
            Default set to "Model running".

        Returns:
            The fitted model.
        """
        self.model = self._compile_model()  # start the model
        # set training steps to use all training data in each epoch
        # TODO: is above description correct?
        steps_per_epoch = self._n_train // self._batch_size
        # set val steps to number of samples, i.e validate images individually
        # TODO: is above description correct?
        validation_steps = self._n_val / self._batch_size
        # Write the main log
        self._current_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.logging(comment)
        # Preparing the pickling.
        checkpoint_filepath = self._path_log + self._current_time + "/checkpoint"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )
        # Fitting the model
        self._model_history = self.model.fit(
            self.train_batches,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.test_batches,
            callbacks=[model_checkpoint_callback],
        )
        self._loss = self._model_history.history["loss"]
        self._val_loss = self._model_history.history["val_loss"]
        self._accuracy = self._model_history.history["accuracy"]
        self._val_accuracy = self._model_history.history["val_accuracy"]
        # logging performances
        self._local_log(comment)
        self.saving_model_performance()

        # Deleting the model if the val accuracy is worse than existing model.
        max_perf = max(self._dictionary_performance.values())
        if max_perf > max(self._val_accuracy):
            for filename in glob.glob(checkpoint_filepath + "*"):
                os.remove(filename)
        else:
            self.logging_saved_model()

        return self._model_history

    def _compile_model(self):
        """
        Takes the Unet models, and compile the model. Return the model.

        Returns:
            the compiled model, with the ADAM gradient descent, the sparse categorical entropy
            and the accuracy metric.
        """
        model = self._setup_unet_model(output_channels=self.output_classes)
        model.compile(
            optimizer="adam",
            loss=BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def _setup_unet_model(self, output_channels: int):
        """
        Unet model from the segmentation notebook by tensorflow. Define the model.

        Args:
        output_channels (int): number of categories in the classification.

        Returns:
            The model unet.

        """
        # initiate the base model
        base_model = self._get_base_model()

        # select the requested down stack layers
        selected_output_layers = [
            base_model.get_layer(name).output for name in self.layers
        ]

        # define the input layer
        inputs = tf.keras.layers.Input(tf.TensorShape(self.input_shape))

        # Downsampling through the model
        # needs to have base model defined.
        down_stack = tf.keras.Model(
            inputs=base_model.input, outputs=selected_output_layers
        )
        # freeze the downstack layers
        down_stack.trainable = False

        # TODO: what does this do and what is 'x'?
        skips = down_stack(inputs)
        layer = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        # TODO: what does this do and why do we need it?
        up_stack = [
            self._upsample(512, 3),  # 4x4 -> 8x8
            self._upsample(256, 3),  # 8x8 -> 16x16
            self._upsample(128, 3),  # 16x16 -> 32x32
            self._upsample(64, 3),  # 32x32 -> 64x64
        ]

        # TODO: how does this work?
        for up, skip in zip(up_stack, skips):
            layer = up(layer)
            concat = tf.keras.layers.Concatenate()
            layer = concat([layer, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            filters=output_channels,
            kernel_size=3,
            strides=2,
            padding="same",
        )  # 64x64 -> 128x128

        layer = last(layer)

        return tf.keras.Model(inputs=inputs, outputs=layer)

    def _get_base_model(self):
        """
        Define the base of the model, MobileNetV2. Note that a discussion on the shape
        is necessary: if the shape of the pictures is the default shape (224, 224, 3),
        the include top options needs to be set to True, and the input shape is not passed
        as an argument of the base model. The default input shape is used.

        Returns:
            the base model MobileNetV2
        """
        if self.input_shape == (224, 224, 3):
            base_model = tf.keras.applications.MobileNetV2(
                include_top=True,
            )
        else:
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
            )
        return base_model

    def _upsample(self, filters, size, apply_dropout=False):
        """Define the upsampling stack. Introduced as an alternative to the pix2pix
           implementation of the tensoflow notebook.
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
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def _display(self, display_list):
        """Display side by side the pictures contained in the list.

        Args:
            a list of three images, aerial pictures, true mask, and predicted mask in that order.

        """
        plt.figure(figsize=(15, 15))
        title = ["Input Image", "True Mask", "Predicted Mask"]
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            type(display_list[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis("off")
        plt.show()

    def _create_mask(self, pred_mask):
        """Create a mask from the predicted array.

        Args:
            a predicted image, through the predict method.

        Returns:
            a mask.

        """
        pred_mask = (pred_mask > 0.5).astype(int) * 255
        return pred_mask

    def show_predictions(self, dataset=None, num=1):
        """Display side by side an earial photography, its true mask, and the predicted mask.

        Args:
            A dataset in the form provided by the dataloader.

        """
        if dataset == None:
            dataset = self.test_batches

        for image_batch, mask_batch in dataset.take(num):
            pred_mask_batch = self.model.predict(image_batch)
            for image, mask, pred_mask in zip(image_batch, mask_batch, pred_mask_batch):
                print(image.shape, mask.shape, pred_mask.shape)
                self._display([image, mask, self._create_mask(pred_mask)])

    def logging(self, comment: str):
        """
        Log the model in the main log.

        Args:
            comment (str): take a string, containing the necessary comment on the model.
        """

        if not os.path.exists(self._path_log):
            os.mkdir(self._path_log)

        main_log = open(self._path_main_log_file, "a")
        main_log.write("\n")
        main_log.write("------")
        main_log.write("\n")
        main_log.write(self._current_time)
        main_log.write("\n")
        main_log.write(comment)
        main_log.write("\n")
        main_log.write("\n")
        main_log.close()

    def _local_log(self, comment: str):
        """
        Create the local log.

        Args:
            comment (str): take a string, containing the necessary comment on the model.
        """

        if not os.path.exists(self._path_log + self._current_time):
            os.mkdir(self._path_log)

        path_local_log = self._path_log + self._current_time + "/local_log.log"
        local_log = open(path_local_log, "a")
        local_log.write(self._current_time)
        local_log.write("\n")
        local_log.write(comment)
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
        local_log.close()

        plt.plot(
            self._model_history.epoch, self._accuracy, "r", label="Training accuracy"
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
        plt.plot(self._model_history.epoch, self._loss, "r", label="Training loss")
        plt.plot(
            self._model_history.epoch, self._val_loss, "bo", label="Validation loss"
        )
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.ylim([0, 1])
        plt.legend()
        path_graph = self._path_log + self._current_time + "/losses.pdf"
        plt.savefig(path_graph)
        plt.close()

    def saving_model_performance(self):
        """save the performance (accuracy) of the model. Print these
        performance in an auxiliary files.
        """
        aux_file = open(self._path_aux, "a")
        aux_file.write(f"{self._current_time} : {max(self._val_accuracy)}")
        aux_file.write("\n")
        aux_file.close()

        with open(self._path_aux) as aux_file:
            for line in aux_file:
                (key, val) = line.split(":")
                self._dictionary_performance[key] = float(val)

    def logging_saved_model(self):
        """Log in the main file tha the model has been saved."""
        main_log = open(self._path_main_log_file, "a")
        main_log.write("Model saved!")
        main_log.write("\n")
        main_log.write(f"Validation accuracy: {max(self._val_accuracy)}")
        main_log.close()
