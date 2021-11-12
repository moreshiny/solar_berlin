import glob
import os
import tensorflow


class DataLoader:
    """Class for creating tensorflow dataset."""

    def __init__(self, path: str, batch_size: int = 32, n_samples: int = None) -> None:
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

        # data has not been curated and wrongly sized images are discarded
        useable_paths = self._discard_wrong_img_paths(all_paths)
        useable_paths.sort()

        # keep only part of image paths if n_samples was specified
        if self.n_samples is None:
            self.n_samples = len(useable_paths)

        assert self.n_samples <= len(
            useable_paths
        ), f"n_samples ({self.n_samples}) is greater than number of available/useable images."
        useable_paths = useable_paths[: self.n_samples]

        # split input and target
        input_paths = [
            filename for filename in useable_paths if "map" in filename]
        target_paths = [
            filename for filename in useable_paths if "mask" in filename]

        assert len(input_paths) == len(
            target_paths
        ), f"Number of input images ({len(input_paths)}) does not match\
                 number of target images ({len(target_paths)})."

        return input_paths, target_paths

    def _discard_wrong_img_paths(self, all_paths):
        """Discard wrong files; function is temporary."""
        correct_filenames = []
        for path in all_paths:
            if tensorflow.keras.utils.load_img(path).size == (224, 224):
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

        [img_rgba, ] = tensorflow.py_function(
            _decode_tensor_load_image, [tensor, "rgba"], [tensorflow.float32]
        )

        # normalize and keep queried channels
        img_rgba = tensorflow.math.divide(img_rgba, 255.0)
        if channels == "RGB":
            img = img_rgba[:, :, :3]
        elif channels == "A":
            img = tensorflow.reshape(tensorflow.math.ceil(
                img_rgba[:, :, 3]), (224, 224, 1))
        # TODO: Double check the line above is correct. Why are there
        #       non-zero-one values in the alpha channel?
        else:
            raise ValueError("Unkown channels specified. Use 'RGB' or 'A'.")
        return img

    def load(self, buffer_size: int = None) -> None:
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
        input_images = []
        for item in self._dataset_input:
            input_images.append(
                self._load_image(item.numpy().decode("utf-8"), "RGB")
            )
        inputs = tensorflow.data.Dataset.from_tensor_slices(input_images)

        target_images = []
        for item in self._dataset_target:
            target_images.append(
                self._load_image(item.numpy().decode("utf-8"), "A")
            )
        targets = tensorflow.data.Dataset.from_tensor_slices(target_images)

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
