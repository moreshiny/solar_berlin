import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow

from unet.unet_resnet_morphological import Unet

OUTPUT_CLASSES = 5  # number of categorical claÔsses.
INPUT_SHAPE = (512, 512, 3)  # input size

model = Unet(
    output_classes=OUTPUT_CLASSES,
    input_shape=INPUT_SHAPE,
    drop_out=False,
    drop_out_rate={"512": 0.3, "256": 0.4, "128": 0.45, "64": 0.5},
    multiclass=bool(OUTPUT_CLASSES - 1),
)

PATH_CHECKPOINT = "logs/03_04_2022_22_51_10/checkpoint.ckpt"

model.load_weights(PATH_CHECKPOINT)
print("loaded")

PATH_TO_PREDICT = "data/benchmark_dataset/test"


all_paths = glob.glob(os.path.join(PATH_TO_PREDICT, "*.tif"))
all_paths += glob.glob(os.path.join(PATH_TO_PREDICT, "*.png"))

all_paths.sort()

input_paths = [filename for filename in all_paths if "map" in filename]
target_paths = [
    filename for filename in all_paths if "mask" in filename or "msk" in filename
]

scce = tensorflow.keras.losses.SparseCategoricalCrossentropy()
cat_accuracy = tensorflow.keras.metrics.CategoricalCrossentropy(OUTPUT_CLASSES)
meaniou5 = tensorflow.keras.metrics.MeanIoU(OUTPUT_CLASSES)
meaniou2 = tensorflow.keras.metrics.MeanIoU(2)
binary_accuracy = tensorflow.keras.metrics.BinaryAccuracy()
sparse_categorical_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(
    name="sparse_categorical_accuracy", dtype=None
)
recall = tensorflow.keras.metrics.Recall(name="recall")
precision = tensorflow.keras.metrics.Precision(name="precision")


list_predictions = []

list_paths = list(zip(input_paths, target_paths))
total = len(list_paths)
for path_image, path_target in tqdm(list_paths, total = total):
    list_info = []
    list_info.append(path_image)
    list_info.append(path_target)
    path_predict = path_image.replace("map", "predict")
    list_info.append(path_predict)
    image = Image.open(path_image)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255
    mask = Image.open(path_target)
    mask = np.array(mask)
    mask = np.ceil(mask * 4 / 255).astype(float)
    pred_prob = model.predict(image)
    pred_prob = pred_prob.squeeze()
    pred = np.argmax(pred_prob, axis=-1).astype(float)
    cat_loss = scce(mask, pred_prob).numpy()
    list_info.append(cat_loss)
    cat_accuracy = sparse_categorical_accuracy(mask, pred_prob).numpy()
    list_info.append(cat_accuracy)
    meaniou_5 = meaniou5(mask, pred).numpy()
    list_info.append(meaniou_5)
    pred_bin = (pred > 0).astype(int)
    mask_bin = (mask > 0).astype(int)
    bin_roof = binary_accuracy(mask_bin, pred_bin).numpy()
    mean_iou_2 = meaniou2(mask_bin, pred_bin).numpy()
    recall_roof = recall(mask_bin, pred_bin).numpy()
    precision_roof = precision(mask_bin, pred_bin).numpy()
    list_info.append(bin_roof)
    list_info.append(mean_iou_2)
    list_info.append(recall_roof)
    list_info.append(precision_roof)
    pred = np.floor(255 / 4 * pred).astype(np.uint8)
    pred_image = Image.fromarray(pred).convert("L")
    pred_image.save(path_predict)
    list_predictions.append(list_info)


columns = [
    "path_image",
    "path_target",
    "path_predict",
    "scce",
    "cat_accuracy",
    "mean_iou_5",
    "bin_acc_roof",
    "mean_iou_2",
    "recall",
    "precision",
]
df_predict = pd.DataFrame(list_predictions, columns=columns)

PATH_TO_CSV = PATH_TO_PREDICT + "/df_predictions.csv"
df_predict.to_csv(PATH_TO_CSV, index=False, header=True)
