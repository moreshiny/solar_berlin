# Logging the models: a parametrization for logs location, tensorboard usage.
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from datetime import datetime


class Logs:
    def __init__(self):
        """
        Initialisation of the class.
        """

        # Paths for the logs
        self._path_log = "logs/"  # Log directory
        # Checking the existence of the log directory
        if not os.path.exists(self._path_log):
            os.mkdir(self._path_log)

        self.path_main_log_file = self._path_log + "main_log.log"  # main log file
        self.path_aux = self._path_log + "log.aux"  # auxiliary log file

        # Preparing the local logs:
        self._current_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        # model checkpoint path
        self.checkpoint_filepath = (
            self._path_log + self._current_time + "/checkpoint.ckpt"
        )
        # tensorboard path
        self.tensorboard_path = self._path_log + "tensorboard/" + self._current_time

    def comment(self, comment: str = "test run") -> None:
        """Writing the experience report from the main log.
        Args:
            comment: a string detailing the experience which is run. Default to "test run".
        """

        with open(self.path_main_log_file, "a", encoding="utf-8") as main_log:
            main_log.write("\n")
            main_log.write("------")
            main_log.write("\n")
            main_log.write(self._current_time)
            main_log.write("\n")
            main_log.write(comment)
            main_log.write("\n")
            main_log.write("\n")

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
        path_snapshot = self._path_log + self._current_time + "/snapshots"
        if not os.path.exists(path_snapshot):
            os.mkdir(path_snapshot)
        path_fig = path_snapshot + f"/output{np.random.rand()}.pdf"
        plt.savefig(path_fig)
        plt.close()
