# This class can be used to the automation of the data cleaning using
#  a efficient image segmentation model.
#
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
        multiclass: bool = False,
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

    def cleaning(self, proportion: float = 0.2):
        """Perform the cleaning according to the losses.
        Args:
        proportion: a float indicating the proportion of element we want to consider for cleaning.

        """

        self._logging_losses()
        n_predict = self._losses.shape[0]
        n_discard = np.ceil(proportion * n_predict).astype(int)
        self._discard_df = self._losses.nlargest(n_discard, "loss")
        self._discard_df["read"] = 0
        self._discard_df["discard"] = 0
        self._discard_df.to_csv(
            self._path_to_clean + "/high_loss_elements.csv", index=False, header=True
        )

        self.bad_images = list(self._discard_df["path_img"])
        self.bad_masks = list(self._discard_df["path_mask"])

    def move_discarded_files(
        self,
        output_folder_name: str = "dirty",
        delete_existing_output_path_no_warning=False,
    ):
        """Copy image files from the input path to the output path.

        Args:
            image_files (list): Filenames as returned by select_random_map_images
            output_path (str): file location to copy to, dafault to "dirty".
            delete_existing_output_path_no_warning (bool, optional): Delete output
            path first if it exists, without warning. Defaults to False.
        """
        if not os.path.exists(self._path_to_clean + "/high_loss_elements.csv"):
            return print(
                "Please provide a CSV file high_loss_elements.csv for cleaning"
            )
        else:
            self._discard_df = pd.read_csv(
                self._path_to_clean + "/high_loss_elements.csv"
            )
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

        mask_discard = self._discard_df["discard"] == 1

        file_to_copy = list(self._discard_df.loc[mask_discard, "path_img"]) + list(
            self._discard_df.loc[mask_discard, "path_mask"]
        )

        for file_path_in in file_to_copy:
            file_path_out = os.path.join(output_path, os.path.basename(file_path_in))
            shutil.move(file_path_in, file_path_out)

    def manual_sorting(self) -> None:
        """
        Trigger the manual sorting of the considered folder.
        Args:
            class_called: a boolean, if the classd has been called before, the method uses
             the parameters of the class, else the method requires an input path.
            path__to_clean: path to be cleaned if class called if false.
        """

        if not os.path.exists(self._path_to_clean + "/high_loss_elements.csv"):

            self._discard_df = pd.DataFrame(columns=["path_img", "path_mask", "loss"])
            self._discard_df["path_img"] = self._input_paths
            self._discard_df["path_mask"] = self._target_paths
            self._discard_df["discard"] = 0
            self._discard_df["read"] = 0

        else:
            self._discard_df = pd.read_csv(
                self._path_to_clean + "/high_loss_elements.csv"
            )

        image_path = ""
        target_path = ""
        break_out_flag = True

        def onpress(event):
            nonlocal image_path, target_path, break_out_flag
            sys.stdout.flush()
            if event.key == "d":
                mask_path = self._discard_df["path_img"] == image_path
                self._discard_df.loc[mask_path, "discard"] = 1
                break_out_flag = False
                plt.close()

            elif event.key == "k":
                mask_path = self._discard_df["path_img"] == image_path
                self._discard_df.loc[mask_path, "discard"] = 0
                break_out_flag = False
                plt.close()

            elif event.key == "q":
                break_out_flag = True

        mask_unread = self._discard_df["read"] == 0
        read_path = self._discard_df.loc[mask_unread].loc[:, ["path_img", "path_mask"]]

        for image_path, target_path in zip(
            read_path["path_img"], read_path["path_mask"]
        ):
            mask_path = self._discard_df["path_img"] == image_path
            self._discard_df.loc[mask_path, "read"] = 1
            img = Image.open(image_path)
            mask = Image.open(target_path)
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 7))
            fig.canvas.mpl_connect("key_press_event", onpress)
            axs[0].imshow(img)
            axs[0].axis("off")
            axs[1].imshow(mask)
            axs[1].axis("off")
            plt.suptitle("Press k to keep, d to discard, q to quit",fontsize=20)
            plt.show()
            if break_out_flag:
                break

        self._discard_df.to_csv(
            self._path_to_clean + "/high_loss_elements.csv", index=False, header=True
        )

        mask_discarded = self._discard_df["discard"] == 1
        self.discard_list = self._discard_df.loc[mask_discarded].loc[
            :, ["path_img", "path_mask"]
        ]

        return self.discard_list
