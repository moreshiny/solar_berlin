import random
import datetime
import cv2
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator

from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo

import os

random.seed(42)

data_dir = "data/"

register_coco_instances("my_dataset_train", {},
                        "data/selected/selected_tiles_512_40000_10000_42/train/coco.json", "data/selected/selected_tiles_512_40000_10000_42/train")
register_coco_instances("my_dataset_val", {},
                        "data/selected/selected_tiles_512_40000_10000_42/test/coco.json", "data/selected/selected_tiles_512_40000_10000_42/test")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

# n_samples divided by batch_size so once per epoch
n_samples = 40000
cfg.SOLVER.IMS_PER_BATCH = 64
cfg.TEST.EVAL_PERIOD = n_samples // cfg.SOLVER.IMS_PER_BATCH

cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.BASE_LR = 0.001  # default LR

# epochs is MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES
# MAX_ITER = epochs * TOTAL_NUM_IMAGES / BATCH_SIZE
epochs = 1
cfg.SOLVER.MAX_ITER = epochs * n_samples // cfg.SOLVER.IMS_PER_BATCH
# cfg.SOLVER.MAX_ITER = 10
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # default

# one class is roof, no roof
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.INPUT.RANDOM_FLIP = "none"  # do not flip
tile_size = 512
cfg.INPUT.MIN_SIZE_TRAIN = tile_size  # keep size as tile_size
cfg.INPUT.MAX_SIZE_TRAIN = tile_size  # keep size as tile_size
cfg.INPUT.MIN_SIZE_TEST = tile_size  # keep size as tile_size
cfg.INPUT.MAX_SIZE_TEST = tile_size  # keep size as tile_size

cfg.INPUT.FORMAT = "RGB"
# (ImageNet RGB instead of BGR)
cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53]
cfg.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]
cfg.INPUT.CROP.ENABLED = False  # do not crop

cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False  # do not skip empty masks

dict_train = DatasetCatalog.get("my_dataset_train")
meta_train = MetadataCatalog.get("my_dataset_train")

minute = datetime.datetime.now().minute
hour = datetime.datetime.now().hour
day = datetime.datetime.now().day
month = datetime.datetime.now().month
year = datetime.datetime.now().year

cfg.OUTPUT_DIR = f"logs/output-{year}-{month}-{day}-{hour}-{minute}"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

image = dict_train[0]
img = cv2.imread(image["file_name"])
visualizer = Visualizer(img, metadata=meta_train)
vis = visualizer.draw_dataset_dict(image)
cv2.imwrite(
    f"{cfg.OUTPUT_DIR}/visualise_train.png", vis.get_image())

os.makedirs(os.path.join(cfg.OUTPUT_DIR, "visualise"), exist_ok=True)
for i, image in enumerate(dict_train):
    img = cv2.imread(image["file_name"])
    visualizer = Visualizer(img, metadata=meta_train)
    vis = visualizer.draw_dataset_dict(image)
    output_image_fn = f"{os.path.join(cfg.OUTPUT_DIR, 'visualise')}/visualise_{image['image_id']}.png"
    cv2.imwrite(output_image_fn, vis.get_image())
    if i > 20:
        break


class EvaluateTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


trainer = EvaluateTrainer(cfg)
trainer.resume_or_load(resume=False)
print(cfg)
trainer.train()

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
# path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

predictor = DefaultPredictor(cfg)

val_metadata = MetadataCatalog.get("my_dataset_val")
dict_val = DatasetCatalog.get("my_dataset_val")

for image in random.sample(dict_val, 100):
    im = cv2.imread(image['file_name'])
    outputs = predictor(im)
    vis1 = Visualizer(
        im,
        metadata=val_metadata,
    )
    out1 = vis1.draw_instance_predictions(outputs["instances"].to("cpu"))

    im2 = cv2.imread(image['file_name'])
    vis2 = Visualizer(
        im2,
        metadata=val_metadata,
    )
    out2 = vis2.draw_dataset_dict(image)

    plt.figure(figsize=(30, 30))
    title = ["Predicted", "Ground Truth"]
    display_list = [out1, out2]
    for j in range(len(display_list)):
        plt.subplot(1, len(display_list), j + 1)
        plt.title(title[j])
        type(display_list[j])
        plt.imshow(display_list[j].get_image())
        plt.axis("off")

    plt.savefig(os.path.join(cfg.OUTPUT_DIR,
                f"prediction-{image['file_name']}"))
