import PIL
import glob
import os
import tensorflow
import tensorflow_io as tfio


class DataLoader:
    def __init__(
        self, path, batch_size=32, input_shape=[224, 224, 3], target_shape=[224, 224, 1]
    ):

        self.path = path
        self.batch_size = batch_size
        self.n_samples = None
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.dataset_input = None
        self.dataset_target = None
        self.dataset = None

        self._initialize_dataset_paths()

    def _initialize_dataset_paths(self):

        img_paths, target_paths = self._get_img_paths()

        self.n_samples = len(img_paths)

        self.dataset_input = tensorflow.data.Dataset.from_tensor_slices(img_paths)
        self.dataset_target = tensorflow.data.Dataset.from_tensor_slices(target_paths)

    def _get_img_paths(self):

        all_paths = glob.glob(os.path.join(self.path, "*.tif"))
        # if n_imgs:
        #     all_paths = all_paths[:n_imgs]

        good_paths = self._discard_wrong_img_paths(all_paths)

        input_paths = [filename for filename in good_paths if "map" in filename]
        target_paths = [filename for filename in good_paths if "mask" in filename]

        input_paths.sort()
        target_paths.sort()

        return input_paths, target_paths

    def _discard_wrong_img_paths(self, all_paths):
        correct_filenames = []
        for path in all_paths:
            if PIL.Image.open(path).size == (224, 224):
                correct_filenames.append(path)
        return correct_filenames

    def _load_images_input(self, path):

        import pdb; pdb.set_trace()
        img_file = tensorflow.io.read_file(path)
        img_rgba = tfio.experimental.image.decode_tiff(img_file)
        img_rgb = tfio.experimental.color.rgba_to_rgb(img_rgba)
        converted_img = tensorflow.image.convert_image_dtype(
            img_rgb, tensorflow.float32
        )
        resized_img = tensorflow.image.resize(converted_img, self.input_shape)

        print(resized_img.shape)
        print(self.input_shape)

        return resized_img

    def _load_images_target(self, path):

        img_file = tensorflow.io.read_file(path)
        img_rgba = tfio.experimental.image.decode_tiff(img_file)
        img_bw = img_rgba[:, :, 3]
        converted_img = tensorflow.image.convert_image_dtype(img_bw, tensorflow.float32)
        resized_img = tensorflow.image.resize(converted_img, self.target_shape)

        print(resized_img.shape)
        print(self.target_shape)

        return resized_img

    def load(self, buffer_size=None):

        if not buffer_size:
            buffer_size = self.n_samples

        inputs = self.dataset_input.map(
            self._load_images_input,
            num_parallel_calls=tensorflow.data.experimental.AUTOTUNE,
        )
        targets = self.dataset_target.map(
            self._load_images_target,
            num_parallel_calls=tensorflow.data.experimental.AUTOTUNE,
        )

        self.dataset = tensorflow.data.Dataset.zip((inputs, targets))

        self.dataset = self.dataset.cache()

        # Shuffle data and create batches
        self.dataset = self.dataset.shuffle(buffer_size=buffer_size)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(self.batch_size)

        # Make dataset fetch batches in the background during the training of the model.
        self.dataset = self.dataset.prefetch(
            buffer_size=tensorflow.data.experimental.AUTOTUNE
        )





    # def to_dataset(self, n_imgs=None):
    #     map_img_paths, mask_img_paths = self._get_img_paths(n_imgs=n_imgs)

    #     dataset = tensorflow.data.Dataset.from_tensor_slices(
    #         (map_img_paths, mask_img_paths)
    #     )

    #     def decode_img(img):

    #         decoded_img = tfio.experimental.image.decode_tiff(img)
    #         print(decoded_img)
    #         return decoded_img

    #     def img_path_to_tensor(map_img_path, mask_img_path):
    #         print(map_img_path)
    #         map_img = tensorflow.io.read_file(map_img_path)
    #         print(map_img)
    #         mask_img = tensorflow.io.read_file(mask_img_path)
    #         decoded_map_img = decode_img(map_img)
    #         decoded_mask_img = decode_img(mask_img)
    #         return (decoded_map_img, decoded_mask_img)

    #     dataset = dataset.map(
    #         img_path_to_tensor, num_parallel_calls=tensorflow.data.AUTOTUNE
    #     )

    #     return dataset

    # def _transform_map_img(self, imgs):
    #     imgs = np.array(PIL.Image.open(imgs))
    #     imgs = tensorflow.cast(imgs, tensorflow.float32) / 255.0
    #     return imgs

    # def _transform_mask_img(self, imgs):
    #     imgs = (np.array(PIL.Image.open(imgs).convert("1")) + 1) / 2
    #     imgs = np.expand_dims(imgs, axis=2)
    #     imgs = tensorflow.cast(imgs, tensorflow.float32)
    #     return imgs
