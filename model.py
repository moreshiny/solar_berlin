"""Define the UNET model. BAsed in part on the image segmentation notebook
https://www.tensorflow.org/tutorials/images/segmentation
"""

import tensorflow as tf
import matplotlib.pyplot as plt

# from IPython.display import clear_output use in the display call back.
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from dataloader import DataLoader


class Model(DataLoader):
    """
    The purpose of the class is to define the Unet model,
    and contain the necessary functions (compile, and fit) to run it.
    """

    def __init__(
        self,
        path_train,
        path_test,
        layer_names,
        output_classes=2,
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
        dl_train = DataLoader(path_train)
        dl_val = DataLoader(path_test)
        dl_train.load()
        dl_val.load()
        self.train_batches = dl_train.dataset
        self.test_batches = dl_val.dataset

        # save size of train and validation data for internal use
        self._n_train = dl_train.n_samples
        self._n_val = dl_val.n_samples
        self._batch_size = dl_train.batch_size

    def model_history(self):
        """
        Train the model. Return the history of the model.

        Returns:
            The fitted model.
        """
        model = self._compile_model()  # start the model

        # set training steps to use all training data in each epoch
        # TODO: is above description correct?
        steps_per_epoch = self._n_train // self._batch_size
        # set val steps to number of samples, i.e validate images individually
        # TODO: is above description correct?
        validation_steps = self._n_val
        model_history = model.fit(
            self.train_batches,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.test_batches,
            # callbacks=[DisplayCallback()],
        )
        return model_history

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
            loss=SparseCategoricalCrossentropy(from_logits=True),
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
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]

    def show_predictions(self, dataset=None, num=1):
        """Display side by side an earial photography, its true mask, and the predicted mask.

        Args:
            A dataset in the form provided by the dataloader.

        """

        model = self.model_history()
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            self._display([image[0], mask[0], self._create_mask(pred_mask)])
