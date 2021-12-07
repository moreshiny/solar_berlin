# Logging the models: a parametrization for logs location, tensorboard usage.
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pprint
import tensorflow
from datetime import datetime


class Logs:
    def __init__(
        self,
        custom_path: str = "",
    ) -> None:
        """
        Initialisation of the class.
        Args:
            local_path: a string defining the name of the loacal path in the form "logs/custom_path"; Default to "".

        """
        # saving the passed variables
        self._custom_path = custom_path

        # Paths for the logs
        self._path_log = "logs/"  # Log directory
        # Checking the existence of the log directory
        if not os.path.exists(self._path_log):
            os.mkdir(self._path_log)

        self.path_main_log_file = self._path_log + "main_log.log"  # main log file
        self.path_aux = self._path_log + "log.aux"  # auxiliary log file

        # Preparing the local logs:
        self._current_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        # Defining the local path
        if self._custom_path == "":
            self._local_path = self._path_log + self._current_time
        else:
            self._local_path = self._path_log + self._custom_path

        # Path to local log file
        self.path_local_log_file = self._local_path + "/local_log.log"

        # model checkpoint path
        self.checkpoint_filepath = self._local_path + "/checkpoint.ckpt"
        # tensorboard path
        self.tensorboard_path = self._path_log + "tensorboard/" + self._current_time

        # local variables
        self._model_config = {}
        self._comment = ""

    def main_log(self, comment: str = "test run", model_config: dict = {}) -> None:
        """Writing the experience report from the main log.
        Args:
            comment: a string detailing the experience which is run. Default to "test run".
            model_config: a dictionary obtained though the get_confi for the model. default to {}.
        """
        self._comment = comment
        self._model_config = model_config
        with open(self.path_main_log_file, "a", encoding="utf-8") as main_log:
            main_log.write("\n")
            main_log.write("------")
            main_log.write("\n")
            main_log.write(self._current_time)
            main_log.write("\n")
            main_log.write(self._comment)
            main_log.write("\n")
            main_log.write(f"Configuration dictionary: {self._model_config}")
            main_log.write("\n")

    def local_log(
        self,
        comment: str = "test run",
        train_data_config: dict = {},
        val_data_config: dict = {},
    ) -> None:
        """Write the local with the characteristics of the model and the used data for the training and validation
        Args:
            train_data_config: a dictionary of train data obtained from the get_config() function of the dataloader\
                 Default to {}.
            val_data_config: a dictionary of val data obtained from the get_config() function of the dataloader\
            Default to {}.
        """
        if not os.path.exists(self._local_path):
            os.mkdir(self._local_path)

        with open(self.path_local_log_file, "a", encoding="utf-8") as local_log:
            local_log.write(self._current_time)
            local_log.write("\n")
            local_log.write(self._comment)
            local_log.write("\n")
            local_log.write(f"Configuration dictionary: {self._model_config}")
            local_log.write("\n")
            local_log.write(f"Train configuration: {train_data_config}")
            local_log.write("\n")
            local_log.write(f"Val configuration: {val_data_config}")

    def show_predictions(
        self,
        dataset: tensorflow.data.Dataset = None,
        model: tensorflow.keras.Model = None,
        num_batches: int = 1,
    ) -> None:
        """Display side by side an earial photography, its true mask, and the
            predicted mask.

        Args:
            A dataset in the form provided by the dataloader.
            A tensorflow keras model.
            num_batches (int): number of batches to display.

        """
        for image_batch, mask_batch in dataset.take(num_batches):
            pred_mask_batch = model.predict(image_batch)
            for image, mask, pred_mask in zip(image_batch, mask_batch, pred_mask_batch):
                self._display([image, mask, self._create_mask(pred_mask)])

    def _create_mask(self, pred_mask):
        """Create a mask from the predicted array.

        Args:
            a predicted image, through the predict method.

        Returns:
            a mask.

        """
        pred_mask = (pred_mask > 0.5).astype(int) * 255
        return pred_mask

    def _display(self, display_list: List) -> None:
        """Display side by side the pictures contained in the list.

        Args:
            display_list (List): Three images, aerial pictures, true mask,
                                and predicted mask in that order.

        """
        plt.figure(figsize=(15, 8))
        title = ["Input Image", "True Mask", "Predicted Mask"]
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            type(display_list[i])
            plt.imshow(tensorflow.keras.utils.array_to_img(display_list[i]))
            plt.axis("off")
        path_snapshot = self._local_path + "/snapshots"
        if not os.path.exists(path_snapshot):
            os.mkdir(path_snapshot)
        path_fig = path_snapshot + f"/output{np.random.rand()}.pdf"
        plt.savefig(path_fig)
        plt.close()
