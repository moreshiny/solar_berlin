import tensorflow as tf

# internal methods needs an underscore as a marker.

class Model():

    def __init__(
        self,
        dataset_train,
        dataset_val,
        layer_names,
        output_classes=2,
        input_shape=[224,224,3],
        include_top=True,
        epochs=10,
        # val_subplits=5, #Check here what this does.
        batch_size=128,
        buffer_size=1000,
    ) -> None:
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.output_classes = output_classes
        self.layers = layer_names
        self.input_shape = input_shape
        self.include_top = include_top
        self.epochs = epochs
        self.train_length = len(dataset_train)
        self.batch_size = batch_size
        self.validation_steps = len(dataset_val)

        self.buffer_size = buffer_size
        self.train_batches = None
        self.test_batches = None
        self.base_model = None
        self.base_model_outputs = None
        ###

    def batches(self):
        """
        """
        if self.batch_size > self.train_length\
             or self.batch_size > len(self.validation_steps):
            self.batch_size = 1
            print("Warning: batch size set to 1!")
        train_batches = (
            self.dataset_train
            .shuffle(self.buffer_size)
            .batch(self.batch_size)
            .repeat()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

        test_batches = (
            self.dataset_val
            .shuffle(self.buffer_size)
            .batch(self.batch_size)
            .repeat()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            )
        return train_batches, test_batches
        ###

    def upsample(self, filters, size, apply_dropout=False):
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

    def initialize_model(self):
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
        self.base_model = base_model
        ###

    def layering(self):
        """
        """
        self.base_model_outputs = [self.base_model.get_layer(
            name).output for name in self.layers]
        ###

    def down_stack(self):
        # Create the feature extraction model

        down_stack = tf.keras.Model(inputs=self.base_model.input,
                                    outputs=self.base_model_outputs)
        down_stack.trainable = False

        return down_stack

    def up_stack(self):
        up_stack = [
            self.upsample(512, 3),  # 4x4 -> 8x8
            self.upsample(256, 3),  # 8x8 -> 16x16
            self.upsample(128, 3),  # 16x16 -> 32x32
            self.upsample(64, 3),   # 32x32 -> 64x64
        ]
        return up_stack

    def unet_model(self, output_channels: int):

        self.initialize_model()  # initiate the model
        self.layering()  # instantiate the layers
        inputs = tf.keras.layers.Input(
            tf.TensorShape(self.input_shape))

    # Downsampling through the model
        down_stack = self.down_stack()  # needs to have base model defined.
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

   # Upsampling and establishing the skip connections
        up_stack = self.up_stack()
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

    def model_compiling(self):
        """
        """
        model = self.unet_model(output_channels=self.output_classes)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])
        return model

    def model_history(self):
        """
        
        """
        train_batches, test_batches = self.batches() 
        print(train_batches, test_batches)  # instantiate the batches.
        model = self.model_compiling()  # start the model

        steps_per_epoch = self.train_length // self.batch_size
        model_history = model.fit(train_batches,
                                  epochs=self.epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=self.validation_steps,
                                  validation_data=test_batches,
                                  # callbacks=[DisplayCallback()]
                                  )
        return model_history
