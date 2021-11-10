import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np


class Model():

    def __init__(
        self,
        dataset_train,
        dataset_val,
        layer_names,
        output_classes=2,
        input_shape=(224, 224, 3),
        include_top=True,
        epochs=10,
        # val_subplits=5, #TODO: Check here what this does.
        batch_size=128,
        buffer_size=1000,
    ) -> None:
        # save the unet model parameters
        self.output_classes = output_classes  # 2 classes for binary classification
        self.epochs = epochs
        # save the layer information for the unet model
        # TODO: can we infer include_top from input_shape?
        self.include_top = include_top  # default True, use the default input layer
        self.input_shape = input_shape  # default for RGB images size 224
        self.layers = layer_names  # layers to be used in the up stack

        # save size of train and validation data for internal use
        self._n_train = len(dataset_train)
        self._n_val = len(dataset_val)

        # store the batch_size or something smaller if it is too large
        self.batch_size = self._safe_batch_size(batch_size)

        # save the training and validation data as batches
        # TODO: move batching to DataLoader?
        self.train_batches = self._batch_convert(dataset_train, buffer_size)
        self.test_batches = self._batch_convert(dataset_val, buffer_size)

    def model_history(self):
        """
        """
        model = self._compile_model()  # start the model

        # set training steps to use all training data in each epoch
        # TODO: is above description correct?
        steps_per_epoch = self._n_train // self.batch_size
        # set val steps to number of samples, i.e validate images individually
        # TODO: is above description correct?
        validation_steps = self._n_val
        model_history = model.fit(self.train_batches,
                                  epochs=self.epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps,
                                  validation_data=self.test_batches,
                                  #TODO: callbacks=[DisplayCallback()]
                                  )
        return model_history

    def _compile_model(self):
        """
        """
        model = self._setup_unet_model(output_channels=self.output_classes)

        model.compile(
            optimizer='adam',
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
        return model

    def _setup_unet_model(self, output_channels: int):
        #TODO: need to credit where we got this?

        # initiate the base model
        base_model = self._get_base_model()

        # select the requested down stack layers
        selected_output_layers =\
            [base_model.get_layer(name).output for name in self.layers]

        # define the input layer
        inputs = tf.keras.layers.Input(tf.TensorShape(self.input_shape))

        # Downsampling through the model
        # needs to have base model defined.
        down_stack = tf.keras.Model(
            inputs=base_model.input,
            outputs=selected_output_layers
        )
        # freeze the downstack layers
        down_stack.trainable = False

        # TODO: what does this do and what is 'x'?
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        # TODO: what does this do and why do we need it?
        up_stack = [
            self._upsample(512, 3),  # 4x4 -> 8x8
            self._upsample(256, 3),  # 8x8 -> 16x16
            self._upsample(128, 3),  # 16x16 -> 32x32
            self._upsample(64, 3),   # 32x32 -> 64x64
        ]

        # TODO: how does this work?
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            filters=output_channels,
            kernel_size=3,
            strides=2,
            padding='same',
        )  # 64x64 -> 128x128

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def _get_base_model(self):
        """
        """
        if self.include_top:
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
        #TODO: need to credit where we got this?
        """
        """
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        )

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def _batch_convert(self, dataset_train, shuffle_buffer_size):
        return (
            dataset_train
            .shuffle(shuffle_buffer_size)
            .batch(self.batch_size)
            .repeat()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    def _safe_batch_size(self, batch_size):
        # TODO a batch_size of less than the length of the datasets does not work
        # figure out why and/or maybe move this to DataLoader

        # determine the smallest of the three values
        safe_batch_size = np.min(
            [self._n_train, self._n_val, batch_size]
        )
        if safe_batch_size < batch_size:
            print(
                f"Warning: batch_size ({batch_size}) is smaller than train or validation data size!")
            print(f"Setting batch_size to {safe_batch_size}.")
        return safe_batch_size
