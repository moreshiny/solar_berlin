# Define the Unet Model with a pretrained Resnet101v2 in the bottom in the for of a keras class.
import tensorflow
from typing import List, Tuple


class Unet(tensorflow.keras.Model):
    """Define the unet model with a pretrained Resnet in the bottom. The Resnet is trained on imagenet, the top layer is removed by default."""

    def __init__(
        self,
        output_classes: int = 1,
        drop_out: bool = False,
        drop_out_rate: dict = {"512": 0, "256": 0, "128": 0, "64": 0},
    ):
        """Class initialisation:
        Args:
            output_classes: number of categorical classes. Default to one.
            drop_out: boolean, wether the dropout in the upstack of the model is activated; Default to False.
            drop_out_rate: If drop_out, defines the dropout rate in the up stacks. Defaults to {"512": 0, "256": 0, "128": 0, "64": 0}
            fine_tune_at: if non zero, freeze the upstacks, and unfreeze the corresponding number of layers in bottom of the pretrained network.
        """
        super(Unet, self).__init__()

        # save the passed variable.
        self._output_classes = output_classes
        self._drop_out = drop_out
        self._drop_out_rate = drop_out_rate

        # call the pretrained model. Trained
        self._base_model = tensorflow.keras.applications.resnet_v2.ResNet101V2(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            # input_shape=self.input_shape,
            pooling=max,
        )
        self._base_model.trainable = False

        # Define the layers for the skip connections.
        # Define first the layers for skip conenctions within the down stacl/pretrained networks
        self._layers = [
            "conv1_conv",
            "conv2_block2_out",
            "conv3_block3_out",
            "conv4_block22_out",
            "conv5_block2_out",
        ]
        self._selected_output_layers = [
            self._base_model.get_layer(name).output for name in self._layers
        ]

        # Define the Unet downstacks.
        self._down_stack = tensorflow.keras.Model(
            inputs=self._base_model.input,
            outputs=self._selected_output_layers,
        )

        # Parameters of the up-stacks.
        FILTERS = [512, 256, 128, 64]
        SIZE = 3

        # Define the up_stacks.
        self._up_stack = []
        for filter in FILTERS:
            upsample = Upsample(
                filter=filter,
                size=SIZE,
                apply_drop_out=self._drop_out,
                drop_out_rate=self._drop_out_rate[f"{filter}"],
            )
            self._up_stack.append(upsample)

        # Last convolution layers.
        self._last_conv = tensorflow.keras.layers.Conv2DTranspose(
            filters=self._output_classes,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="sigmoid",
        )

    @tensorflow.function
    def call(self, input):
        """
        Build the Unet model.
        Args:
            Input: A 4D-tensors

        """

        # building the downstacks
        skips = self._down_stack(input)
        layer = skips[-1]
        skips = reversed(skips[:-1])

        # upsampling and establishing the skip connections
        drop_out = self._drop_out

        # Building the skip connections.
        for up, skip in zip(self._up_stack, skips):
            layer = up(layer)
            concat = tensorflow.keras.layers.Concatenate()
            layer = concat([layer, skip])
        # Last layers
        layer = self._last_conv(layer)

        return layer

    def freezing_layers(self, fine_tune_at: int = 0) -> None:
        """
        When called, freeze the up-stacks, and unfreeze some of the top layers of the pretrained model.
        Args:
            fine_tune_at: number of layers of the pretrained model which are unfrozen.Default to 0.

        """
        self._base_model.trainable = True

        for i, layer in enumerate(self._base_model.layers[0:-fine_tune_at]):
            layer.trainable = False

        for i, layer in enumerate(self._base_model.layers[-fine_tune_at:]):
            layer.trainable = True

        for stack in self._up_stack:
            stack.trainable = False

    def get_config(self):
        """Overwrite the get_config() methods to save and load the model.
        see the documentation:
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        """
        return {
            "output_classes": self._output_classes,
            "drop_out": self._drop_out,
            "drop_out_rate": self._drop_out_rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Upsample(tensorflow.keras.Model):
    """Define the upstack black used in the Unet."""

    def __init__(
        self,
        filter,
        size,
        apply_drop_out: bool = False,
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
            drop_out_rate: If apply_dropout, defines the droptout rate.

        Returns:
               Upsample Sequential Model
        """
        super(Upsample, self).__init__()
        self._filter = filter
        self._size = size
        self._drop_out = apply_drop_out
        self._drop_out_rate = drop_out_rate

        self.sequential = tensorflow.keras.Sequential()

        initializer = tensorflow.random_normal_initializer(0.0, 0.02)

        self.conv_transpose = tensorflow.keras.layers.Conv2DTranspose(
            filters=self._filter,
            kernel_size=self._size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )

        self.normalization = tensorflow.keras.layers.BatchNormalization()

        self.dropout_layer = tensorflow.keras.layers.Dropout(drop_out_rate)

        self.activation_layer = tensorflow.keras.layers.ReLU()

    @tensorflow.function
    def call(self, input):
        """Build the model."""

        x = self.sequential(input)
        x = self.conv_transpose(x)
        x = self.normalization(x)
        # conditional dropout layers
        if self._drop_out:
            x = self.dropout_layer(x)

        x = self.activation_layer(x)
        return x

    def get_config(self):
        """Overwrite the get_config() methods to save and load the model."""
        return {
            "filter": self._filter,
            "size": self._size,
            "apply_drop_out": self._drop_out,
            "drop_out_rate": self._drop_out_rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
