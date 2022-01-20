"""Define the Unet Model with a pretrained Resnet101v2
in the bottom in the for of a keras class."""
from typing import Tuple, List
import tensorflow


class Unet(tensorflow.keras.Model):
    """Define the unet model with a pretrained Resnet in the bottom.\
        The Resnet is trained on imagenet, the top layer is removed by default."""

    def __init__(
        self,
        output_classes: int = 1,
        input_shape: Tuple = (512, 512, 3),
        drop_out: bool = False,
        drop_out_rate: dict = {"512": 0, "256": 0, "128": 0, "64": 0},
        fine_tune_at: int = 0,
        upstack_trainable: bool = True,
        multiclass: bool = False,
    ):
        """Class initialisation:
        Args:
            output_classes: number of categorical classes. Default to one.
            input_shape: Define the size of the input and output of the model.
            Defaut to (512,512, 3).
            resizing_shape: Size of the output resizing layer.
            drop_out: boolean, wether the dropout in the upstack of the model is activated;
            Default to False.
            drop_out_rate: If drop_out, defines the dropout rate in the up stacks.
            Defaults to {"512": 0, "256": 0, "128": 0, "64": 0}
            fine_tune_at: if non zero, freeze the upstacks, and unfreeze the corresponding
            number of layers in bottom of the pretrained network. Default to 0.
            upstack_trainable: Boolean, if True, the upstack is trainable. Default to True.
            multiclass: boolean, if True, modify the activation of the last convolution to softmax,\
                 else sigmoid. Default to False.
        """
        super(Unet, self).__init__()

        # save the passed variable.
        self._output_classes = output_classes
        self._input_shape = input_shape
        self._drop_out = drop_out
        self._drop_out_rate = drop_out_rate
        self._fine_tune_at = fine_tune_at
        self._upstack_trainable = upstack_trainable

        # Input layer

        self._input_layer = tensorflow.keras.layers.InputLayer(self._input_shape)

        # Define the layers for the skip connections.
        # Define first the layers for skip conenctions within the down stacl/pretrained networks
        self._layers = [
            "conv1_conv",
            "conv2_block2_out",
            "conv3_block3_out",
            "conv4_block22_out",
            "conv5_block2_out",
        ]

        # Define the Unet downstacks.

        self._down_stack = Downsample(self._layers, self._fine_tune_at)

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
            upsample.trainable = self._upstack_trainable
            self._up_stack.append(upsample)

        # Concatenate layer
        self.concatenate = tensorflow.keras.layers.Concatenate()

        # Last convolution layers.
        if multiclass:
            activation = "softmax"
        else:
            activation = "sigmoid"
        self._last_conv = tensorflow.keras.layers.Conv2DTranspose(
            filters=self._output_classes,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=activation,
        )

        self._last_conv.trainable = self._upstack_trainable

    @tensorflow.function
    def call(self, input):
        """
        Build the Unet model.
        Args:
            Input: A 4D-keras-tensor

        """

        # building the downstacks

        layer = self._input_layer(input)

        skips = self._down_stack(layer)
        layer = skips[-1]
        skips = reversed(skips[:-1])

        # Building the skip connections.
        for up, skip in zip(self._up_stack, skips):
            layer = up(layer)
            layer = self.concatenate([layer, skip])
        # Last convolution
        layer = self._last_conv(layer)

        return layer

    def get_config(self):
        """Overwrite the get_config() methods to save and load the model.
        see the documentation:
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        """
        return {
            "output_classes": self._output_classes,
            "input_shape": self._input_shape,
            "drop_out": self._drop_out,
            "drop_out_rate": self._drop_out_rate,
            "fine_tune_at": self._fine_tune_at,
            "upstack_trainable": self._upstack_trainable,
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
        """Class function which can be used to define a model from a configuration dictionary.
        Args:
            config: a configuration dictionary in the format of the get_config function.

        """
        return cls(**config)


class Downsample(tensorflow.keras.Model):
    """Define the downstacks of the unet Model."""

    def __init__(
        self,
        layer_names: List = [],
        fine_tune_at: int = 0,
    ) -> None:
        """Class initialisation:
        Args:
            layer_names: list of strings containing the layer names
            which defines the skip connections. Default to [].
            trainable_layer: number of frozen layer at teh head of the pretrained network.
        """
        super(Downsample, self).__init__()
        # Saving the layer names.

        self._layers = layer_names
        self._fine_tune_at = fine_tune_at

        # Calling the base model.
        self._base_model = tensorflow.keras.applications.resnet_v2.ResNet101V2(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            pooling=max,
        )
        # setting which layers of the downstack are trainable.
        if fine_tune_at > 0:
            self._base_model.trainable = True
            for _, layer in enumerate(self._base_model.layers[0 : -2 * fine_tune_at]):
                layer.trainable = False
            for _, layer in enumerate(self._base_model.layers[-2 * fine_tune_at :]):
                layer.trainable = True
        else:
            self._base_model.trainable = False

        # Listing the output of the model.
        self._selected_output_layers = [
            self._base_model.get_layer(name).output for name in self._layers
        ]

        self._down_stack = tensorflow.keras.Model(
            inputs=self._base_model.input,
            outputs=self._selected_output_layers,
        )

    @tensorflow.function
    def call(self, input):
        """Build the model:
        Args:
            input: a 4d tf Tensor.

        """
        return self._down_stack(input)

    def get_config(self):
        """Overwrite the get_config() methods to save and load the model."""
        return {"layer_names": self._layers, "fine_tune_at": self._fine_tune_at}

    @classmethod
    def from_config(cls, config):
        """Class function which can be used to define a model from a configuration dictionary.
        Args:
            config: a configuration dictionary in the format of the get_config function.

        """
        return cls(**config)
