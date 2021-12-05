# Logging the models: a parametrization for logs location, tensorboard usage.
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD

=======
>>>>>>> bd0fe62 (updated model class (layer included), added logging class, updated the training. Explanation on how to use pickle model is added)
import tensorflow
from datetime import datetime


class Logs:
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
    def __init__(self):
=======
    def __init__(self, model: tensorflow.keras.Model = None) -> None:
>>>>>>> e83f518 (updated model class (layer included), added logging class, updated the training. Explanation on how to use pickle model is added)
        """
        Initialisation of the class.
        Args:
            model: a tensofrlow keras model.
        """
<<<<<<< HEAD
>>>>>>> bd0fe62 (updated model class (layer included), added logging class, updated the training. Explanation on how to use pickle model is added)
=======
        # saving the passed variables
        self._model = model
>>>>>>> e83f518 (updated model class (layer included), added logging class, updated the training. Explanation on how to use pickle model is added)

        # Paths for the logs
        self._path_log = "logs/"  # Log directory
        # Checking the existence of the log directory
        if not os.path.exists(self._path_log):
            os.mkdir(self._path_log)

        self.path_main_log_file = self._path_log + "main_log.log"  # main log file
        self.path_aux = self._path_log + "log.aux"  # auxiliary log file

        # Preparing the local logs:
        self._current_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

<<<<<<< HEAD
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
=======
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

>>>>>>> bd0fe62 (updated model class (layer included), added logging class, updated the training. Explanation on how to use pickle model is added)
        with open(self.path_main_log_file, "a", encoding="utf-8") as main_log:
            main_log.write("\n")
            main_log.write("------")
            main_log.write("\n")
            main_log.write(self._current_time)
            main_log.write("\n")
<<<<<<< HEAD
            main_log.write(self._comment)
            main_log.write("\n")
            main_log.write(f"Configuration dictionary: {self._model_config}")
            main_log.write("\n")

    def local_log(
        self,
        train_data_config: dict = {},
        val_data_config: dict = {},
        metrics: dict = {},
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
            local_log.write(f"Model configuration: {self._model_config}")
            local_log.write("\n")
            local_log.write(f"Train configuration: {train_data_config}")
            local_log.write("\n")
            local_log.write(f"Train configuration: {val_data_config}")
            local_log.write("\n")
            for key, values in metrics.items():
                local_log.write(f"{key}: {values}")
                local_log.write("\n")

        len_dict = len(metrics)

        plt.figure(figsize=(8, 4 * len_dict))

        position = 1
        for key, values in metrics.items():
            plt.subplot(len_dict, 1, position)
            max_v = max(values[0] + values[1])
            min_v = min(values[0] + values[1])
            plt.plot(values[0], label=f"Training {key}")
            plt.plot(values[1], label=f"Validation {key}")
            plt.ylim([0.7 * min_v, 1.2 * max_v])
            plt.legend(bbox_to_anchor=(0.05, 1.05), loc="upper left")
            plt.title(f"Training and Validation {key}")
            position += 1

        plt.xlabel("epoch")

        path_graph = self._path_log + self._current_time + "/losses.pdf"
        plt.savefig(path_graph)
        plt.close()

=======
            main_log.write(comment)
            main_log.write("\n")
            main_log.write(f"Configuration dictionary: {self._model.get_config()}")
            main_log.write("\n")

>>>>>>> bd0fe62 (updated model class (layer included), added logging class, updated the training. Explanation on how to use pickle model is added)
    def show_predictions(
        self,
        dataset: tensorflow.data.Dataset = None,
        num_batches: int = 1,
<<<<<<< HEAD
        multiclass: bool = False,
=======
>>>>>>> bd0fe62 (updated model class (layer included), added logging class, updated the training. Explanation on how to use pickle model is added)
    ) -> None:
        """Display side by side an earial photography, its true mask, and the
            predicted mask.

        Args:
            A dataset in the form provided by the dataloader.
            A tensorflow keras model.
            num_batches (int): number of batches to display.
<<<<<<< HEAD
            multiclass (bool): if True, activate the calculation of the mask\
                 prediction for multiclass problems.
=======
>>>>>>> bd0fe62 (updated model class (layer included), added logging class, updated the training. Explanation on how to use pickle model is added)

        """
        for image_batch, mask_batch in dataset.take(num_batches):
            pred_mask_batch = self._model.predict(image_batch)
            for image, mask, pred_mask in zip(image_batch, mask_batch, pred_mask_batch):
<<<<<<< HEAD
                self._display(
                    [image, mask, self._create_mask(pred_mask, multiclass=multiclass)]
                )

    def _create_mask(self, pred_mask, multiclass: bool = False):
        """Create a mask from the predicted array.

        Args:
            pred_mask: a predicted mask, through the predict method.
            multiclass: A boolean. If true, activate the multiclass calculation of for the calculation of the mask

=======
                self._display([image, mask, self._create_mask(pred_mask)])

    def _create_mask(self, pred_mask):
        """Create a mask from the predicted array.

        Args:
            a predicted image, through the predict method.
>>>>>>> bd0fe62 (updated model class (layer included), added logging class, updated the training. Explanation on how to use pickle model is added)

        Returns:
            a mask.

        """
<<<<<<< HEAD
        if multiclass:
            pred_mask = np.argmax(pred_mask, axis=-1)
            pred_mask = np.expand_dims(pred_mask, axis=-1)
        else:
            pred_mask = (pred_mask > 0.5).astype(int) * 255

=======
        pred_mask = (pred_mask > 0.5).astype(int) * 255
>>>>>>> bd0fe62 (updated model class (layer included), added logging class, updated the training. Explanation on how to use pickle model is added)
        return pred_mask

    def _display(self, display_list: List) -> None:
        """Display side by side the pictures contained in the list.

        Args:
            display_list (List): Three images, aerial pictures, true mask,
                                and predicted mask in that order.

        """
<<<<<<< HEAD
        path_snapshot = self._local_path + "/snapshots"
        if not os.path.exists(path_snapshot):
            os.mkdir(path_snapshot)
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        type(display_list[0])
        plt.imshow(tensorflow.keras.utils.array_to_img(display_list[0]))
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        type(display_list[1])
        plt.imshow(tensorflow.keras.utils.array_to_img(display_list[1]), cmap="plasma")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(display_list[2], cmap="plasma")
        plt.axis("off")
=======
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
>>>>>>> bd0fe62 (updated model class (layer included), added logging class, updated the training. Explanation on how to use pickle model is added)
        path_fig = path_snapshot + f"/output{np.random.rand()}.pdf"
        plt.savefig(path_fig)
        plt.close()
