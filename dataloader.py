import PIL
import glob
import os
import numpy as np
import tensorflow

class DataLoader:

    def __init__(self, path):
        self.path = path

    def to_dataset(self, n_imgs=-1):
        map_imgs, mask_imgs = self._get_imgs(n_imgs=n_imgs)

        map_train = []
        mask_train = []

        for map_img, mask_img in zip(map_imgs, mask_imgs):
            map_img = self._transform_map_img(map_img)
            mask_img = self._transform_mask_img(mask_img)
            map_train.append(map_img)
            mask_train.append(mask_img)

        dataset = tensorflow.data.Dataset.from_tensor_slices((map_train, mask_train))

        return dataset

    def _discard_wrong_img_paths(self, all_img_paths):
        correct_filenames = []
        for path in all_img_paths:
            if PIL.Image.open(path).size == (224, 224):
                correct_filenames.append(path)
        return correct_filenames

    def _get_imgs(self, n_imgs):
        
        all_img_paths = glob.glob(os.path.join(self.path, "*.tif"))
        all_img_paths = all_img_paths[:n_imgs]

        good_img_paths = self._discard_wrong_img_paths(all_img_paths)

        map_imgs = [filename for filename in good_img_paths if "map" in filename]
        mask_imgs = [filename for filename in good_img_paths if "mask" in filename]

        map_imgs.sort()
        mask_imgs.sort()

        return map_imgs, mask_imgs

    def _transform_map_img(self, imgs):
        imgs = np.array(PIL.Image.open(imgs))
        imgs = tensorflow.cast(imgs, tensorflow.float32) / 255.0
        return imgs

    def _transform_mask_img(self, imgs):
        imgs = (np.array(PIL.Image.open(imgs).convert("1")) + 1)/2
        imgs = np.expand_dims(imgs, axis=2)
        imgs = tensorflow.cast(imgs, tensorflow.float32)
        return imgs
