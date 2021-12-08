import numpy as np
import glob
import os
import tensorflow


class DataLoader:
    """Class for creating tensorflow dataset."""

    def __init__(
        self,
        path: str,
        batch_size: int = 32,
        n_samples: int = None,
        input_shape: tuple = (224, 224, 3),
        multiclass: bool = False,
        legacy_mode: bool = True,
    ) -> None:
        """Class instance initialization.

        Args:
            path (str): Path to data folder with "map" and "mask" pairs.
            batch_size (int, optional): Batch size for model training.
                Defaults to 32.
            n_samples(int, optional): Number of input-target pairs to load.
                Returns all available pairs if set to None. Defaults to None.
        """
        # initialize attributes
        self.path = path
        self.batch_size = batch_size
        self.n_samples = n_samples
        self._dataset_input = None
        self._dataset_target = None
        self.dataset = None
        self.input_shape = input_shape

        if legacy_mode:
            assert (
                multiclass == False
            ), "Legacy mode is not compatible with multiclass mode."

        self._legacy_mode = legacy_mode
        self._multiclass = multiclass

        # TODO remove legacy mode when no longer needed
        if legacy_mode:
            assert (
                multiclass == False
            ), "Legacy mode is not compatible with multiclass mode."
        self._legacy_mode = legacy_mode
        self._multiclass = multiclass

        # initialize self.dataset_input and self.dataset_target
        self._initialize_dataset_paths()

    def _initialize_dataset_paths(self) -> None:
        """Initialize self.dataset_input and self.dataset_target as datasets
        that contain file path names with input and target images.
        """
        # get image paths
        img_paths, target_paths = self._get_img_paths()

        # create datasets
        self._dataset_input = tensorflow.data.Dataset.from_tensor_slices(img_paths)
        self._dataset_target = tensorflow.data.Dataset.from_tensor_slices(target_paths)

    def _get_img_paths(self) -> tuple:
        """Retrieves all image paths for input and targets present in
        self.path and sorts the resulting lists.

        Returns:
            tuple: A tuple of style (input_paths, target_paths) where
                both input_paths and target_paths are sorted lists with file
                path names to .tif images. The Nth element of target_paths
                corresponds to the target of the Nth element of input_paths.
        """
        # get all paths
        all_paths = glob.glob(os.path.join(self.path, "*.tif"))
        all_paths += glob.glob(os.path.join(self.path, "*.png"))

        # data has not been curated and wrongly sized images are discarded
        useable_paths = self._discard_wrong_img_paths(all_paths)
        useable_paths.sort()

        # keep only part of image paths if n_samples was specified
        if self.n_samples is None:
            self.n_samples = len(useable_paths) // 2

        # we need to get twice as many paths as requested samples (map and mask)
        n_paths = self.n_samples * 2

        assert n_paths <= len(
            useable_paths
        ), f"""n_samples ({self.n_samples}) is greater than number of
                available/useable images {len(useable_paths) // 2}."""

        # keep only the first n_paths paths
        useable_paths = useable_paths[:n_paths]

        # split input and target
        input_paths = [filename for filename in useable_paths if "map" in filename]
        # TODO "mask" is needed only for legacy mode, remove when no longer needed
        target_paths = [
            filename
            for filename in useable_paths
            if "mask" in filename or "msk" in filename
        ]

        assert len(input_paths) == len(
            target_paths
        ), f"""Number of input images ({len(input_paths)}) does not match
                number of target images ({len(target_paths)})."""

        return input_paths, target_paths

    def _discard_wrong_img_paths(self, all_paths):
        """Discard wrong files; function is temporary."""
        correct_filenames = []
        for path in all_paths:
            if tensorflow.keras.utils.load_img(path).size == self.input_shape[:2]:
                correct_filenames.append(path)
        return correct_filenames

    def _load_image(self, tensor: tensorflow.Tensor, channels: str) -> tensorflow.image:
        """Load .tif image as tensor and perform normalization.

        Args:
            tensor (tensorflow.Tensor): Tensor with file name path of .tif
                image to be loaded.
            channels (str): The color channels of the image that is desired.
                Either "RGB" or "A" (alpha). All channels are normalized such
                that 255 = 1.0.

        Returns:
            tensorflow.image: An image tensor of type float32 normalized from
                0-255 to 0.0-1.0.
        """
        # decode tensor and read image using helper function
        def _decode_tensor_load_image(tensor, color_mode):
            img = tensorflow.keras.utils.load_img(
                tensor.numpy().decode("utf-8"),
                color_mode=color_mode,
            )
            return img

        if channels == "RGB" or channels == "A":
            [image,] = tensorflow.py_function(
                _decode_tensor_load_image, [tensor, "rgba"], [tensorflow.float32]
            )
        elif channels == "L":
            [image,] = tensorflow.py_function(
                _decode_tensor_load_image, [tensor, "grayscale"], [tensorflow.float32]
            )

        # normalize and keep queried channels
        image = tensorflow.math.divide(image, 255.0)
        if channels == "RGB":
            img = image[:, :, :3]
        elif channels == "A":
            # for mask images - legacy mode
            # legacy images are (partially) transparent in non-roof areas
            # take ceiling of transparancy so "no roof" is 1 and "roof" is 0
            img = tensorflow.math.ceil(image[:, :, 3])
            img = tensorflow.reshape(img, self.input_shape[:2] + tuple([1]))
        elif channels == "L":
            # for mask images - grey scale with minimum value for "no roof"
            # higher values indicate better pv suitability categories
            if self._multiclass:
                # normalise image to 0-4 (0 = no roof, 4 = best pv category)
                img = tensorflow.math.multiply(image, 4)
                img = tensorflow.math.ceil(img)
                img = tensorflow.reshape(img, self.input_shape[:2] + tuple([1]))
            else:
                # floor and invert image so "no roof" is 1 and "roof" is 0
                img = tensorflow.math.floor(image)
                img = tensorflow.math.subtract(1.0, img)
                img = tensorflow.reshape(img, self.input_shape[:2] + tuple([1]))
        else:
            raise ValueError("Unkown channels specified. Use 'RGB' or 'A'.")
        return img

    def load(self, buffer_size: int = 500) -> None:
        """Load images into dataset and store in self.dataset attribute. The
        dataset will contain (input, target) pairs. Addional settings actions
        performed on the dataset are:
            .cache()
            .shuffle(buffer_size=buffer_size)
            .repeat()
            .batch(self.batch_size)
            .prefetch()

        Args:
            buffer_size (int, optional): Number of elements from this dataset
                from which the new dataset will sample. If set to None, the
                number of elements will be the full dataset. Defaults to None.
        """
        # if unset, set buffersize to number of samples
        if not buffer_size:
            buffer_size = self.n_samples

        # load images for inputs and targets
        inputs = self._dataset_input.map(
            lambda t: self._load_image(tensor=t, channels="RGB"),
            num_parallel_calls=tensorflow.data.experimental.AUTOTUNE,
        )

        if self._legacy_mode:
            targets = self._dataset_target.map(
                lambda t: self._load_image(tensor=t, channels="A"),
                num_parallel_calls=tensorflow.data.experimental.AUTOTUNE,
            )
        else:
            targets = self._dataset_target.map(
                lambda t: self._load_image(tensor=t, channels="L"),
                num_parallel_calls=tensorflow.data.experimental.AUTOTUNE,
            )

        # store in attribute
        self.dataset = tensorflow.data.Dataset.zip((inputs, targets))

        # caching
        self.dataset = self.dataset.cache()

        # shuffle and create batches
        self.dataset = self.dataset.shuffle(buffer_size=buffer_size)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(self.batch_size)

        # fetch batches in background during model training
        self.dataset = self.dataset.prefetch(
            buffer_size=tensorflow.data.experimental.AUTOTUNE
        )

    def get_config(self) -> dict:
        """Return the key characteristics of the loaded data"""
        return {
            "Data path": self.path,
            "Batch size": self.batch_size,
            "Number of samples": self.n_samples,
            "Shape": self.input_shape,
        }
