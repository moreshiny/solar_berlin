import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
<<<<<<< HEAD

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


metrics = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "IoU": jaccard_score,
    "recall": recall_score,
    "precision": precision_score,
}
=======
import tensorflow


OUTPUT_CLASSES = 5
meaniou5 = tensorflow.keras.metrics.MeanIoU(OUTPUT_CLASSES)
meaniou2 = tensorflow.keras.metrics.MeanIoU(2, name="mean_iou2")
binary_accuracy = tensorflow.keras.metrics.BinaryAccuracy(name="accuracy")
sparse_categorical_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(
    name="sparse_categorical_accuracy", dtype=None
)
recall = tensorflow.keras.metrics.Recall(name="recall")
precision = tensorflow.keras.metrics.Precision(name="precision")

cat_metrics = [meaniou5, sparse_categorical_accuracy]

bin_metrics = [binary_accuracy, meaniou2, recall, precision]
>>>>>>> 4a22009 (script added to calculate and store metrics from a prediction folder)


def cat_accuracy(y_true: np.array, y_pred: np.array) -> float:
    return np.sum(np.equal(y_true, y_pred)) / (512 * 512)


PATH_TO_PREDICT = "data/bin_clean_4000/test_pred"

<<<<<<< HEAD
COLOURS_NAME = [0, 63, 127, 191, 255]
=======
COLOURS_NAME = [63, 127, 191, 255]
>>>>>>> 4a22009 (script added to calculate and store metrics from a prediction folder)

all_paths = glob.glob(os.path.join(PATH_TO_PREDICT, "*.tif"))
all_paths += glob.glob(os.path.join(PATH_TO_PREDICT, "*.png"))

all_paths.sort()

input_paths = [filename for filename in all_paths if "map" in filename]
target_paths = [
    filename for filename in all_paths if "mask" in filename or "msk" in filename
]
predict_paths = [filename for filename in all_paths if "predict" in filename]

<<<<<<< HEAD
print("Paths collected")

=======
>>>>>>> 4a22009 (script added to calculate and store metrics from a prediction folder)
df_predict_no_loss = pd.DataFrame()

df_predict_no_loss["input_paths"] = input_paths
df_predict_no_loss["target_paths"] = target_paths
df_predict_no_loss["predict_paths"] = predict_paths


def pixels_count_per_colour(path, colour):
    AREA = 512 * 512
    mask = Image.open(path)
    mask = np.array(mask)
    mask = mask == colour
    return np.sum(mask) / AREA


for col in ["target", "predict"]:
    for colour in COLOURS_NAME:
        df_predict_no_loss[f"area_{col}_{colour}"] = df_predict_no_loss[
            f"{col}_paths"
        ].apply(lambda x: pixels_count_per_colour(x, colour))

for colour in COLOURS_NAME:
    df_predict_no_loss[f"diff_area_{colour}"] = (
        df_predict_no_loss[f"area_target_{colour}"]
        - df_predict_no_loss[f"area_predict_{colour}"]
<<<<<<< HEAD
    ) / df_predict_no_loss[f"area_target_{colour}"]

df_predict_no_loss = df_predict_no_loss.replace([-np.inf, np.inf], np.nan)
=======
    )
>>>>>>> 4a22009 (script added to calculate and store metrics from a prediction folder)


def open_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(int)
    return image


<<<<<<< HEAD
print("sizes calculated")

=======
>>>>>>> 4a22009 (script added to calculate and store metrics from a prediction folder)
df_predict_no_loss["cat_accuracy"] = [
    cat_accuracy(open_image(path_mask), open_image(path_predict))
    for path_mask, path_predict in zip(
        df_predict_no_loss["target_paths"], df_predict_no_loss["predict_paths"]
    )
]


def normalize(array):
    return np.ceil(4 / 255 * array)


<<<<<<< HEAD
=======
df_predict_no_loss["mean_iou_5"] = [
    meaniou5(
        normalize(open_image(path_mask)), normalize(open_image(path_predict))
    ).numpy()
    for path_mask, path_predict in zip(
        df_predict_no_loss["target_paths"], df_predict_no_loss["predict_paths"]
    )
]


>>>>>>> 4a22009 (script added to calculate and store metrics from a prediction folder)
def bin_mask_roof(array):
    return (array > 0).astype(int)


<<<<<<< HEAD
for key, values in metrics.items():
    df_predict_no_loss[f"roof_{key}"] = [
        values(
            np.ravel(bin_mask_roof(open_image(path_mask))),
            np.ravel(bin_mask_roof(open_image(path_predict))),
        )
=======
for metric in bin_metrics:
    df_predict_no_loss[f"roof_{metric.name}"] = [
        metric(
            bin_mask_roof(open_image(path_mask)),
            bin_mask_roof(open_image(path_predict)),
        ).numpy()
>>>>>>> 4a22009 (script added to calculate and store metrics from a prediction folder)
        for path_mask, path_predict in zip(
            df_predict_no_loss["target_paths"], df_predict_no_loss["predict_paths"]
        )
    ]
<<<<<<< HEAD
    print(f"roof_{key} calculated")


print(f"Binary metrics for roof done")
=======
>>>>>>> 4a22009 (script added to calculate and store metrics from a prediction folder)


def bin_mask(array, colour):
    return (array == colour).astype(int)


for colour in COLOURS_NAME:
<<<<<<< HEAD
    for key, values in metrics.items():
        df_predict_no_loss[f"{key}_{colour}"] = [
            values(
                np.ravel(bin_mask(open_image(path_mask), colour)),
                np.ravel(bin_mask(open_image(path_predict), colour)),
            )
=======
    for metric in bin_metrics:
        df_predict_no_loss[f"{metric.name}_{colour}"] = [
            metric(
                bin_mask(open_image(path_mask), colour),
                bin_mask(open_image(path_predict), colour),
            ).numpy()
>>>>>>> 4a22009 (script added to calculate and store metrics from a prediction folder)
            for path_mask, path_predict in zip(
                df_predict_no_loss["target_paths"], df_predict_no_loss["predict_paths"]
            )
        ]
<<<<<<< HEAD
        print(f"{key}_{colour} calculated")

jaccard = []

for col in COLOURS_NAME:
    jaccard.append(f"IoU_{col}")

df_predict_no_loss["mean_IoU"] = np.mean(df_predict_no_loss[jaccard], axis=1)


print("Dumping the file")
PATH_TO_CSV = PATH_TO_PREDICT + "/df_predictions_no_loss.csv"
df_predict_no_loss.to_csv(PATH_TO_CSV, index=False, header=True)
print("Done")
=======

PATH_TO_CSV = PATH_TO_PREDICT + "/df_predictions_no_loss.csv"
df_predict_no_loss.to_csv(PATH_TO_CSV, index=False, header=True)
>>>>>>> 4a22009 (script added to calculate and store metrics from a prediction folder)
