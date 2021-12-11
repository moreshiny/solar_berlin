# This class can be used to the automation of the data cleaning using
#  a efficient image segmentation model.
import os
import glob
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow
from roof.errors import OutputPathExistsError



class DataCleaning:
    def __init__(
        self,
        path_to_clean: str,
        input_shape: tuple = (512, 512, 3),
        model: tensorflow.keras.models = None,
    ) -> None:
        """ Class instantiation (right word?).
        Args:
            path: a string, path to the folder which will be screened.
            input_shape: Input shape of the images to consider, default to (512, 512, 3).
            model: a Keras model used for instance segmentation of the roof. Default to None.
            accuracy_threshold: an integer defining the threshold at which\
                the accuracy is considered bad and the image needs to be discarded.
        """
        # saving the variables.
        self._path_to_clean = path_to_clean
        self._input_shape = input_shape
        self._model = model

        if not os.path.exists(self._path_to_clean):
            raise OutputPathExistsError(
                "The past you are trying to clean does not exist. Be careful, Jérémie!"
            )
        
        # local variables.
        self.path_images = []
        self.bad_images = []
        self.bad_masks = []
        self._scce = tensorflow.keras.losses.SparseCategoricalCrossentropy()
        self._losses = []
        self._output_path = ""
        self._keep_list = []
        self.discard_list = []
        self._discard_df = None

        self._input_paths, self._target_paths = self._get_img_paths()

    # Code borrowed from the dataloader.
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
        all_paths = glob.glob(os.path.join(self._path_to_clean, "*.tif"))
        all_paths += glob.glob(os.path.join(self._path_to_clean, "*.png"))

        # data has not been curated and wrongly sized images are discarded
        useable_paths = self._discard_wrong_img_paths(all_paths)
        useable_paths.sort()

        # keep only part of image paths if n_samples was specified
        n_samples = len(useable_paths) // 2

        # we need to get twice as many paths as requested samples (map and mask)
        n_paths = n_samples * 2

        assert n_paths <= len(
            useable_paths
        ), f"""n_samples ({n_samples}) is greater than number of
                available/useable images {len(useable_paths) // 2}."""

        # keep only the first n_paths paths
        useable_paths = useable_paths[:n_paths]

        # split input and target
        input_paths = [filename for filename in useable_paths if "map" in filename]
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
            if tensorflow.keras.utils.load_img(path).size == self._input_shape[:2]:
                correct_filenames.append(path)
        return correct_filenames

    def _logging_losses(self):
        """Logs the imqges/mqsk loss qccording to the model."""

        for image_path, mask_path in zip(self._input_paths, self._target_paths):
            img = Image.open(image_path)
            img = img.convert("RGB")
            img = np.array(img)
            img = np.expand_dims(img, axis=0)
            mask = Image.open(mask_path)
            mask = mask.convert("1")
            mask = np.array(mask)
            pred = self._model.predict(img)
            loss = self._scce(mask, pred).numpy()
            list_df = [image_path, mask_path, loss]
            self._losses.append(list_df)

        self._losses = pd.DataFrame(
            self._losses, columns=["path_img", "path_mask", "loss"]
        )

    def cleaning(self, proportion: float = 0.2, out_folder_name: str = "dirty"):
        """Perform the cleaning according to the losses.
        Args:
        proportion: a float indicating the proportion of element we want to consider for cleaning.

        """

        self._logging_losses()
        n_predict = self._losses.shape[0]
        n_discard = np.ceil(proportion * n_predict).astype(int)
        self._discard_df = self._losses.nlargest(n_discard, "loss")

        self.bad_images = list(self._discard_df["path_img"])
        self.bad_masks = list(self._discard_df["path_mask"])

        self._move_bad_files(out_folder_name)

    def _move_bad_files(
        self, output_folder_name: str, delete_existing_output_path_no_warning=False
    ):
        """Copy image files from the input path to the output path.

        Args:
            image_files (list): Filenames as returned by select_random_map_images
            input_path (str): original file location
            output_path (str): file location to copy to
            delete_existing_output_path_no_warning (bool, optional): Delete output
            path first if it exists, without warning. Defaults to False.
        """

        # if we have been asked to, delete existing out path without warning
        # TODO is this safe?

        output_path = self._path_to_clean + "/" + output_folder_name + "/"

        self._output_path = output_path

        if delete_existing_output_path_no_warning and os.path.exists(output_path):
            shutil.rmtree(output_path)

        # create output path if it doesn't exist or end if it does
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        else:
            raise OutputPathExistsError(
                "At least one of the output directory already exists."
                "\nSet delete_existing=True to remove it."
            )

        # get file names into a dict for easier processing

        file_to_copy = self.bad_images + self.bad_masks

        for file_path_in in file_to_copy:
            file_path_out = os.path.join(output_path, os.path.basename(file_path_in))
            shutil.copy(file_path_in, file_path_out)

    def manual_sorting(
        self,
        class_called: bool = True,
        path_to_clean: str = "",
    ) -> None:
        """
        Trigger the manual sorting of the considered folder.
        Args:
            class_called: a boolean, if the classd has been called before, the method uses
             the parameters of the class, else the method requires an input path.
            path__to_clean: path to be cleaned if class called if false.
        """
        
        if not class_called:
            
            all_paths = glob.glob(os.path.join(path_to_clean, "*.tif"))
            all_paths += glob.glob(os.path.join(path_to_clean, "*.png"))

            all_paths.sort()

            input_paths = [filename for filename in all_paths if "map" in filename]
            target_paths = [
                filename
                for filename in all_paths
                if "mask" in filename or "msk" in filename
            ]
            self._keep_list = []
            self.discard_list = []
        else:
            input_paths = list(self._discard_df.loc[:,"path_img"])
            target_paths = list(self._discard_df.loc[:,"path_mask"])
            
        image_path = ""
        target_path = ""
        break_out_flag = True

        def onpress(event):
            nonlocal image_path, target_path, break_out_flag
            sys.stdout.flush()
            if event.key == "d":
                self.discard_list.append(image_path)
                self.discard_list.append(target_path)
                break_out_flag = False
                plt.close()

            elif event.key == "k":
                self._keep_list.append(image_path)
                self._keep_list.append(target_path)
                break_out_flag = False
                plt.close()

            elif event.key == "q":
                break_out_flag = True

        for image_path, target_path in zip(input_paths, target_paths):
            img = Image.open(image_path)
            mask = Image.open(target_path)
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 7))
            plt.title("Press k to keep, d to discard, q to stop")
            fig.canvas.mpl_connect("key_press_event", onpress)
            axs[0].imshow(img)
            axs[0].axis("off")
            axs[1].imshow(mask)
            axs[1].axis("off")
            plt.title("Press k to keep, d to discard, s to stop")
            plt.show()
            if break_out_flag:
                break

        return self.discard_list
