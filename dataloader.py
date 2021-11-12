import PIL
import glob
import os
import tensorflow


class DataLoader:
    """Class for creating tensorflow dataset."""

    def __init__(self, path: str, batch_size: int = 32) -> None:
        """Class instance initialization.

        Args:
            path (str): Path to data folder with "map" and "mask" pairs.
            batch_size (int, optional): Batch size for model training.
                Defaults to 32.
        """
        # initialize attributes
        self.path = path
        self.batch_size = batch_size
        self.dataset_input = None
        self.dataset_target = None
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
        self.dataset_input = tensorflow.data.Dataset.from_tensor_slices(img_paths)
        self.dataset_target = tensorflow.data.Dataset.from_tensor_slices(target_paths)

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
        good_paths = self._discard_wrong_img_paths(all_paths)

        # split input and target
        input_paths = [filename for filename in good_paths if "map" in filename]
        target_paths = [filename for filename in good_paths if "mask" in filename]

        # sort
        input_paths.sort()
        target_paths.sort()

        return input_paths, target_paths

    def _discard_wrong_img_paths(self, all_paths):
        """Discard wrong files; function is temporary."""
        correct_filenames = []
        for path in all_paths:
            if PIL.Image.open(path).size == (224, 224):
                correct_filenames.append(path)
        return correct_filenames

    def _load_image(self, path: str, is_input: bool) -> tensorflow.image:
        """Load .tif image as tensor and perform normalization.

        Args:
            path (str): File name path of .tif image to be loaded.
            is_input (bool): If True the output image keeps its RGB color
                channels and is then normalized to 255=1.0. If False only the
                alpha channel of the RGBA color mode is returned without
                further normalization.

        Returns:
            tensorflow.image: An image tensor of type float32 normalized from
                0-255 to 0.0-1.0.
        """
        # read file and decode
        img_rgba = PIL.Image.open(path).convert("RGBA")
        # convert to float32
        converted_img = tensorflow.image.convert_image_dtype(
            img_rgba, tensorflow.float32
        )

        # keep the right color channels
        if is_input:
            img = converted_img[:, :, :3]  # / 255.0
        else:
            img = tensorflow.reshape(converted_img[:, :, 3], (224, 224, 1))

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
            buffer_size = self.dataset_input.cardinality().numpy()

        # load images for inputs and targets
        input_images = []
        for item in self.dataset_input:
            input_images.append(
                self._load_image(item.numpy().decode("utf-8"), is_input=True)
            )
        inputs = tensorflow.data.Dataset.from_tensor_slices(input_images)

        target_images = []
        for item in self.dataset_target:
            target_images.append(
                self._load_image(item.numpy().decode("utf-8"), is_input=False)
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
