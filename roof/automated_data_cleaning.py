<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> e3c44a6 (model class moved to the model folder, further doctrings added, further cleaning of the folder)
"""This class can be used to the automation of the data cleaning using
  a efficient image segmentation model. The cleaning is based on estimating the
  highest loss of a given keas model, which is passed to the class.
"""
<<<<<<< HEAD
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
    """This class can be used to the automation of the data cleaning using
    a efficient image segmentation model. The cleaning is based on estimating the
    highest loss of a given keas model, which is passed to the class.
    """

=======
# This class can be used to the automation of the data cleaning using a efficient image segmentation model.
=======
# This class can be used to the automation of the data cleaning using
#  a efficient image segmentation model.
<<<<<<< HEAD
>>>>>>> e374b66 (tinder like data cleaning added, as well as a runf file added)
=======
#
>>>>>>> 62c2250 (updated cleaning, and test cleaning class updated: the cleaning can now be started again at all time)
=======
>>>>>>> e3c44a6 (model class moved to the model folder, further doctrings added, further cleaning of the folder)
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
<<<<<<< HEAD
>>>>>>> 55ca74c (data cleaning added, minor change to the training, and dataloader.)
=======
    """This class can be used to the automation of the data cleaning using
    a efficient image segmentation model. The cleaning is based on estimating the
    highest loss of a given keas model, which is passed to the class.
    """

>>>>>>> e3c44a6 (model class moved to the model folder, further doctrings added, further cleaning of the folder)
    def __init__(
        self,
        path_to_clean: str,
        input_shape: tuple = (512, 512, 3),
        model: tensorflow.keras.models = None,
    ) -> None:
<<<<<<< HEAD
<<<<<<< HEAD
        """ Class initialisation.
        Args:
            path: a string, path to the folder which will be screened.
            input_shape: Input shape of the images to consider, default to (512, 512, 3).
            model: a Keras model used for instance segmentation of the roof. Default to None.
=======
        """ Class instantiation (right word?).
=======
        """ Class initialisation.
>>>>>>> e3c44a6 (model class moved to the model folder, further doctrings added, further cleaning of the folder)
        Args:
            path: a string, path to the folder which will be screened.
            input_shape: Input shape of the images to consider, default to (512, 512, 3).
<<<<<<< HEAD
            model: a Keras model used for instance segmentation of the roof. Default to None. 
>>>>>>> 55ca74c (data cleaning added, minor change to the training, and dataloader.)
=======
            model: a Keras model used for instance segmentation of the roof. Default to None.
>>>>>>> e374b66 (tinder like data cleaning added, as well as a runf file added)
            accuracy_threshold: an integer defining the threshold at which\
                the accuracy is considered bad and the image needs to be discarded.
        """
        # saving the variables.
        self._path_to_clean = path_to_clean
        self._input_shape = input_shape
        self._model = model

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> e374b66 (tinder like data cleaning added, as well as a runf file added)
        if not os.path.exists(self._path_to_clean):
            raise OutputPathExistsError(
                "The past you are trying to clean does not exist. Be careful, Jérémie!"
            )
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> 55ca74c (data cleaning added, minor change to the training, and dataloader.)
=======
        
>>>>>>> e374b66 (tinder like data cleaning added, as well as a runf file added)
=======

>>>>>>> 62c2250 (updated cleaning, and test cleaning class updated: the cleaning can now be started again at all time)
        # local variables.
        self.path_images = []
        self.bad_images = []
        self.bad_masks = []
        self._scce = tensorflow.keras.losses.SparseCategoricalCrossentropy()
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> bbd5c40 (automated cleaning class added with automatic discard of blank masks with high losses)
        self._binary_crossentropy = tensorflow.keras.losses.BinaryCrossentropy(
            from_logits=False
        )
        self._losses = []
        self._output_path = ""
        self._keep_list = []
        self.discard_list = []
        self._discard_df = None
=======
 
        self._losses = []
        
>>>>>>> 55ca74c (data cleaning added, minor change to the training, and dataloader.)
=======
        self._losses = []
        self._output_path = ""
        self._keep_list = []
        self.discard_list = []
        self._discard_df = None
>>>>>>> e374b66 (tinder like data cleaning added, as well as a runf file added)

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

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> bbd5c40 (automated cleaning class added with automatic discard of blank masks with high losses)
    def _logging_losses(self, proportion_empty: int = 0.25):
        """Logs the imqges/mqsk loss qccording to the model.
        Args:
            proportion_empty: an integer for selecting images whose masks are at least
            1 - proportion_empty empty. The corresponding images will be automatically
            discarded.

        """
<<<<<<< HEAD
=======
    def _logging_losses(self):
        """Logs the imqges/mqsk loss qccording to the model."""
>>>>>>> 55ca74c (data cleaning added, minor change to the training, and dataloader.)
=======
>>>>>>> bbd5c40 (automated cleaning class added with automatic discard of blank masks with high losses)

        for image_path, mask_path in zip(self._input_paths, self._target_paths):
            img = Image.open(image_path)
            img = img.convert("RGB")
<<<<<<< HEAD
<<<<<<< HEAD
            img = np.array(img) / 255
            img = np.expand_dims(img, axis=0)
            mask = Image.open(mask_path)
            mask = np.array(mask)
            pred = self._model.predict(img)
            pred_bin = 1 - pred[0, :, :, 0]
            pred_bin = pred_bin.squeeze()
            mask_bin = (mask > 0).astype(int)
            mask_bin = mask_bin.squeeze()
            cat_loss = self._scce(mask, pred).numpy()
            bin_loss = self._binary_crossentropy(mask_bin, pred_bin).numpy()
            if (
                np.sum(mask_bin)
                < proportion_empty * self._input_shape[0] * self._input_shape[1]
            ):
                discard = 1
                read = 1
            else:
                discard = 0
                read = 0
            list_df = [image_path, mask_path, cat_loss, bin_loss, discard, read]
            self._losses.append(list_df)

        columns = ["path_img", "path_mask", "cat_loss", "bin_loss", "discard", "read"]
        self._losses = pd.DataFrame(self._losses, columns=columns)

    def cleaning(
        self,
        proportion: float = 0.2,
        proportion_empty: float = 0.25,
        proportion_discarded_empty: float = 0.75,
    ):
        """Perform the cleaning according to the losses.
        Args:
        proportion: a float indicating the proportion of element we want to consider for cleaning.
        proportion_empty: an integer for selecting images whose masks are at least
                1 - proportion_empty empty. The corresponding images will be automatically
                discarded.
        proportion_discarded_empty: a float inidicated the proportion of low-loss empty mask that
                will be discared.

        """

        self._logging_losses(proportion_empty=proportion_empty)
        n_predict = self._losses.shape[0]
        n_discard = np.ceil(proportion * n_predict).astype(int)
        n_discard_empty = np.floor((1 - proportion) * n_predict).astype(int)

        complement_loss = self._losses.nsmallest(n_discard_empty, "bin_loss")
        activated_pixels_list = []
        for path_mask in complement_loss["path_mask"]:
            mask = Image.open(path_mask)
            mask = np.array(mask)
            mask_bin = mask > 0
            if np.sum(mask_bin) < 0.01 * self._input_shape[0] * self._input_shape[1]:
                activated_pixels_list.append(True)
            else:
                activated_pixels_list.append(False)
        complement_loss["almost_empty"] = activated_pixels_list
        mask_empty_mask = complement_loss["almost_empty"] == True
        empty_mask_df = complement_loss.loc[mask_empty_mask, :]
        empty_mask_df = empty_mask_df.sample(frac=proportion_discarded_empty)
        empty_mask_df.drop("almost_empty", axis=1)
        empty_mask_df.loc[:, "discard"] = 1

        self._discard_df = self._losses.nlargest(n_discard, "bin_loss")
        self._discard_df.append(empty_mask_df)

        self._discard_df.to_csv(
            self._path_to_clean + "/high_loss_elements.csv", index=False, header=True
        )

        print("Number of images:", self._losses.shape[0])
        print("Number of images of high losses:", self._discard_df.shape[0])
        n_presorted = np.sum(self._discard_df["discard"])
        print(
            f"High-Loss automatically dicarded based on {proportion_empty}pc empty mask:",
            n_presorted,
        )
        print(
            f"{proportion_discarded_empty}pc Low-loss automatically discarded based",
            empty_mask_df.shape[0],
        )
        print("Images to sort manually:", self._discard_df.shape[0] - n_presorted)

        self.bad_images = list(self._discard_df["path_img"])
        self.bad_masks = list(self._discard_df["path_mask"])

    def move_discarded_files(
        self,
        output_folder_name: str = "dirty",
        delete_existing_output_path_no_warning=False,
    ):
=======
            img = np.array(img)
=======
            img = np.array(img) / 255
>>>>>>> bbd5c40 (automated cleaning class added with automatic discard of blank masks with high losses)
            img = np.expand_dims(img, axis=0)
            mask = Image.open(mask_path)
            mask = np.array(mask)
            pred = self._model.predict(img)
            pred_bin = 1 - pred[0, :, :, 0]
            pred_bin = pred_bin.squeeze()
            mask_bin = (mask > 0).astype(int)
            mask_bin = mask_bin.squeeze()
            cat_loss = self._scce(mask, pred).numpy()
            bin_loss = self._binary_crossentropy(mask_bin, pred_bin).numpy()
            if (
                np.sum(mask_bin)
                < proportion_empty * self._input_shape[0] * self._input_shape[1]
            ):
                discard = 1
                read = 1
            else:
                discard = 0
                read = 0
            list_df = [image_path, mask_path, cat_loss, bin_loss, discard, read]
            self._losses.append(list_df)

        columns = ["path_img", "path_mask", "cat_loss", "bin_loss", "discard", "read"]
        self._losses = pd.DataFrame(self._losses, columns=columns)

    def cleaning(
        self,
        proportion: float = 0.2,
        proportion_empty: float = 0.25,
        proportion_discarded_empty: float = 0.75,
    ):
        """Perform the cleaning according to the losses.
        Args:
        proportion: a float indicating the proportion of element we want to consider for cleaning.
        proportion_empty: an integer for selecting images whose masks are at least
                1 - proportion_empty empty. The corresponding images will be automatically
                discarded.
        proportion_discarded_empty: a float inidicated the proportion of low-loss empty mask that
                will be discared.

        """

        self._logging_losses(proportion_empty=proportion_empty)
        n_predict = self._losses.shape[0]
        n_discard = np.ceil(proportion * n_predict).astype(int)
        n_discard_empty = np.floor((1 - proportion) * n_predict).astype(int)

        complement_loss = self._losses.nsmallest(n_discard_empty, "bin_loss")
        activated_pixels_list = []
        for path_mask in complement_loss["path_mask"]:
            mask = Image.open(path_mask)
            mask = np.array(mask)
            mask_bin = mask > 0
            if np.sum(mask_bin) < 0.01 * self._input_shape[0] * self._input_shape[1]:
                activated_pixels_list.append(True)
            else:
                activated_pixels_list.append(False)
        complement_loss["almost_empty"] = activated_pixels_list
        mask_empty_mask = complement_loss["almost_empty"] == True
        empty_mask_df = complement_loss.loc[mask_empty_mask, :]
        empty_mask_df = empty_mask_df.sample(frac=proportion_discarded_empty)
        empty_mask_df.drop("almost_empty", axis=1)
        empty_mask_df.loc[:, "discard"] = 1

        self._discard_df = self._losses.nlargest(n_discard, "bin_loss")
        self._discard_df.append(empty_mask_df)

        self._discard_df.to_csv(
            self._path_to_clean + "/high_loss_elements.csv", index=False, header=True
        )

        print("Number of images:", self._losses.shape[0])
        print("Number of images of high losses:", self._discard_df.shape[0])
        n_presorted = np.sum(self._discard_df["discard"])
        print(
            f"High-Loss automatically dicarded based on {proportion_empty}pc empty mask:",
            n_presorted,
        )
        print(
            f"{proportion_discarded_empty}pc Low-loss automatically discarded based",
            empty_mask_df.shape[0],
        )
        print("Images to sort manually:", self._discard_df.shape[0] - n_presorted)

        self.bad_images = list(self._discard_df["path_img"])
        self.bad_masks = list(self._discard_df["path_mask"])

<<<<<<< HEAD
        self._move_bad_files(out_folder_name)

<<<<<<< HEAD



    def _move_bad_files(self, output_folder_name: str, delete_existing_output_path_no_warning=False):
>>>>>>> 55ca74c (data cleaning added, minor change to the training, and dataloader.)
=======
    def _move_bad_files(
        self, output_folder_name: str, delete_existing_output_path_no_warning=False
=======
    def move_discarded_files(
        self,
        output_folder_name: str = "dirty",
        delete_existing_output_path_no_warning=False,
>>>>>>> 62c2250 (updated cleaning, and test cleaning class updated: the cleaning can now be started again at all time)
    ):
>>>>>>> e374b66 (tinder like data cleaning added, as well as a runf file added)
        """Copy image files from the input path to the output path.

        Args:
            image_files (list): Filenames as returned by select_random_map_images
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
            input_path (str): original file location
            output_path (str): file location to copy to
            delete_existing_output_path_no_warning (bool, optional): Delete output
            path first if it exists, without warning. Defaults to False.
        """

>>>>>>> 55ca74c (data cleaning added, minor change to the training, and dataloader.)
=======
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
>>>>>>> 62c2250 (updated cleaning, and test cleaning class updated: the cleaning can now be started again at all time)
        # if we have been asked to, delete existing out path without warning
        # TODO is this safe?

        output_path = self._path_to_clean + "/" + output_folder_name + "/"

<<<<<<< HEAD
<<<<<<< HEAD
        self._output_path = output_path

=======
>>>>>>> 55ca74c (data cleaning added, minor change to the training, and dataloader.)
=======
        self._output_path = output_path

>>>>>>> e374b66 (tinder like data cleaning added, as well as a runf file added)
        if delete_existing_output_path_no_warning and os.path.exists(output_path):
            shutil.rmtree(output_path)

        # create output path if it doesn't exist or end if it does
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        else:
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> e374b66 (tinder like data cleaning added, as well as a runf file added)
            raise OutputPathExistsError(
                "At least one of the output directory already exists."
                "\nSet delete_existing=True to remove it."
            )
<<<<<<< HEAD

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

            print("No CSV file read: browsing existing documents")
<<<<<<< HEAD
            self._discard_df = pd.DataFrame(columns=["path_img", "path_mask"])
            self._discard_df["path_img"] = self._input_paths
            self._discard_df["path_mask"] = self._target_paths
            self._discard_df["discard"] = 0
            self._discard_df["read"] = 0

        else:
            print("CSV file found: loading the document.")
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
        mask_not_discarded = self._discard_df["discard"] == 0
        mask_unread_notdiscarded_yet = mask_unread & mask_not_discarded
        read_path = self._discard_df.loc[
            mask_unread_notdiscarded_yet, ["path_img", "path_mask"]
        ]

        to_go = read_path.shape[0]

        print(to_go)

        counter = 0
        for image_path, target_path in zip(
            read_path["path_img"], read_path["path_mask"]
        ):
            counter += 1
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
            plt.suptitle(
                f"Press k to keep, d to discard, q to quit; Remaining images: {to_go - counter}",
                fontsize=20,
            )
            plt.show()
            if break_out_flag:
                break
        self._discard_df.to_csv(
            self._path_to_clean + "/high_loss_elements.csv", index=False, header=True
        )
        mask_discarded = self._discard_df["discard"] == 1
        self.discard_list = self._discard_df.loc[
            mask_discarded, ["path_img", "path_mask"]
        ]
        return self.discard_list
=======
            raise OutputPathExistsError("At least one of the output directory already exists."
                                        "\nSet delete_existing=True to remove it.")
=======
>>>>>>> e374b66 (tinder like data cleaning added, as well as a runf file added)

        # get file names into a dict for easier processing

        mask_discard = self._discard_df["discard"] == 1

        file_to_copy = list(self._discard_df.loc[mask_discard, "path_img"]) + list(
            self._discard_df.loc[mask_discard, "path_mask"]
        )

        for file_path_in in file_to_copy:
            file_path_out = os.path.join(output_path, os.path.basename(file_path_in))
            shutil.move(file_path_in, file_path_out)

<<<<<<< HEAD
<<<<<<< HEAD
        for file_path_in in file_to_move:
            file_path_out = os.path.join(
                        output_path, os.path.basename(file_path_in)
                    )
            shutil.move(file_path_in, file_path_out)
>>>>>>> 55ca74c (data cleaning added, minor change to the training, and dataloader.)
=======
    def manual_sorting(
        self,
        class_called: bool = True,
        path_to_clean: str = "",
    ) -> None:
=======
    def manual_sorting(self) -> None:
>>>>>>> 62c2250 (updated cleaning, and test cleaning class updated: the cleaning can now be started again at all time)
        """
        Trigger the manual sorting of the considered folder.
        Args:
            class_called: a boolean, if the classd has been called before, the method uses
             the parameters of the class, else the method requires an input path.
            path__to_clean: path to be cleaned if class called if false.
        """

        if not os.path.exists(self._path_to_clean + "/high_loss_elements.csv"):

=======
>>>>>>> a90ca2b (updated cleaning class: calculation of remaining updated)
            self._discard_df = pd.DataFrame(columns=["path_img", "path_mask"])
            self._discard_df["path_img"] = self._input_paths
            self._discard_df["path_mask"] = self._target_paths
            self._discard_df["discard"] = 0
            self._discard_df["read"] = 0

        else:
            print("CSV file found: loading the document.")
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
        mask_not_discarded = self._discard_df["discard"] == 0
        mask_unread_notdiscarded_yet = mask_unread & mask_not_discarded
        read_path = self._discard_df.loc[
            mask_unread_notdiscarded_yet, ["path_img", "path_mask"]
        ]

        to_go = read_path.shape[0]

        print(to_go)

        counter = 0
        for image_path, target_path in zip(
            read_path["path_img"], read_path["path_mask"]
        ):
            counter += 1
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
            plt.suptitle(
                f"Press k to keep, d to discard, q to quit; Remaining images: {to_go - counter}",
                fontsize=20,
            )
            plt.show()
            if break_out_flag:
                break
        self._discard_df.to_csv(
            self._path_to_clean + "/high_loss_elements.csv", index=False, header=True
        )
        mask_discarded = self._discard_df["discard"] == 1
        self.discard_list = self._discard_df.loc[
            mask_discarded, ["path_img", "path_mask"]
        ]
        return self.discard_list
>>>>>>> e374b66 (tinder like data cleaning added, as well as a runf file added)
