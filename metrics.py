import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

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


def cat_accuracy(y_true: np.array, y_pred: np.array) -> float:
    return np.sum(np.equal(y_true, y_pred)) / (512 * 512)


PATH_TO_PREDICT = "data/benchmark_dataset/test"

COLOURS_NAME = [0, 63, 127, 191, 255]

all_paths = glob.glob(os.path.join(PATH_TO_PREDICT, "*.tif"))
all_paths += glob.glob(os.path.join(PATH_TO_PREDICT, "*.png"))

all_paths.sort()

input_paths = [filename for filename in all_paths if "map" in filename]
target_paths = [
    filename for filename in all_paths if "mask" in filename or "msk" in filename
]
predict_paths = [filename for filename in all_paths if "predict" in filename]

print("Paths collected")

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
    ) / df_predict_no_loss[f"area_target_{colour}"]

df_predict_no_loss = df_predict_no_loss.replace([-np.inf, np.inf], np.nan)


def open_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(int)
    return image


print("sizes calculated")

df_predict_no_loss["cat_accuracy"] = [
    cat_accuracy(open_image(path_mask), open_image(path_predict))
    for path_mask, path_predict in zip(
        df_predict_no_loss["target_paths"], df_predict_no_loss["predict_paths"]
    )
]


def normalize(array):
    return np.ceil(4 / 255 * array)


def bin_mask_roof(array):
    return (array > 0).astype(int)


for key, values in metrics.items():
    df_predict_no_loss[f"roof_{key}"] = [
        values(
            np.ravel(bin_mask_roof(open_image(path_mask))),
            np.ravel(bin_mask_roof(open_image(path_predict))),
        )
        for path_mask, path_predict in zip(
            df_predict_no_loss["target_paths"], df_predict_no_loss["predict_paths"]
        )
    ]
    print(f"roof_{key} calculated")


print(f"Binary metrics for roof done")


def bin_mask(array, colour):
    return (array == colour).astype(int)


for colour in COLOURS_NAME:
    for key, values in metrics.items():
        df_predict_no_loss[f"{key}_{colour}"] = [
            values(
                np.ravel(bin_mask(open_image(path_mask), colour)),
                np.ravel(bin_mask(open_image(path_predict), colour)),
            )
            for path_mask, path_predict in zip(
                df_predict_no_loss["target_paths"], df_predict_no_loss["predict_paths"]
            )
        ]
        print(f"{key}_{colour} calculated")

jaccard = []

for col in COLOURS_NAME:
    jaccard.append(f"IoU_{col}")

df_predict_no_loss["mean_IoU"] = np.mean(df_predict_no_loss[jaccard], axis=1)


print("Dumping the file")
PATH_TO_CSV = PATH_TO_PREDICT + "/df_predictions_no_loss.csv"
df_predict_no_loss.to_csv(PATH_TO_CSV, index=False, header=True)
print("Done")
