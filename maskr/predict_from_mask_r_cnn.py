import shutil
import glob
import torch
import numpy as np
import random
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2 import model_zoo

import os

random.seed(42)

data_dir = "data/"

register_coco_instances("my_dataset_val", {},
                        "data/selected/selected_tiles_512_100_20_42_binary/train/coco.json", "data/selected/selected_tiles_512_100_20_42_binary/train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TEST = ("my_dataset_val",)

# n_samples divided by batch_size so once per epoch
n_samples = 2000
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.TEST.EVAL_PERIOD = n_samples // cfg.SOLVER.IMS_PER_BATCH

cfg.DATALOADER.NUM_WORKERS = 1
cfg.SOLVER.BASE_LR = 0.001  # default LR

# epochs is MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES
# MAX_ITER = epochs * TOTAL_NUM_IMAGES / BATCH_SIZE
epochs = 8
cfg.SOLVER.MAX_ITER = epochs * n_samples // cfg.SOLVER.IMS_PER_BATCH
# cfg.SOLVER.MAX_ITER = 10
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # default

# one class is roof, no roof
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.INPUT.RANDOM_FLIP = "none"  # do not flip
tile_size = 512
cfg.MODEL.ROI_MASK_HEAD.CONV_DIM = 512
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

cfg.OUTPUT_DIR = "logs/output-2021-12-15-00-24"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
# path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, f"model_0019999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0   # set a custom testing threshold

predictor = DefaultPredictor(cfg)

val_metadata = MetadataCatalog.get("my_dataset_val")
dict_val = DatasetCatalog.get("my_dataset_val")

predictions_dir = os.path.join(cfg.OUTPUT_DIR, "predictions")
bin_mask_dir = os.path.join(cfg.OUTPUT_DIR, "bin_masks")
os.makedirs(predictions_dir, exist_ok=True)
os.makedirs(bin_mask_dir, exist_ok=True)

# for image in random.sample(dict_val, 100):
#     im = cv2.imread(image['file_name'])
#     outputs = predictor(im)
#     vis1 = Visualizer(
#         im,
#         metadata=val_metadata,
#     )
#     out1 = vis1.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imwrite(
#         f"{predictions_dir}/{image['image_id']}_predicted.png", out1.get_image())

#     im2 = cv2.imread(image['file_name'])
#     vis2 = Visualizer(
#         im2,
#         metadata=val_metadata,
#     )
#     out2 = vis2.draw_dataset_dict(image)
#     cv2.imwrite(
#         f"{predictions_dir}/{image['image_id']}_true.png", out2.get_image())


images = glob.glob(
    "data/selected/selected_tiles_512_4000_1000_42/test/*_map.png")
masks = glob.glob(
    "data/selected/selected_tiles_512_4000_1000_42/test/*_msk.png")

output_dir = "data/selected/selected_tiles_512_4000_1000_42_transparent/test/"

os.makedirs(output_dir, exist_ok=True)


# copy all masks to the output dir
for mask in masks:
    shutil.copy(mask, output_dir)

for img_fn in images:

    #for idx, image_dict in enumerate(random.sample(dict_val, 100)):
    #img_fn = image
    img = cv2.imread(img_fn)
    # print(img_fn)
    # print(img)
    outputs = predictor(img)
    scores = outputs["instances"].scores.tolist()
    #print(scores)
    mask = outputs['instances'].get('pred_masks')
    print(mask.shape)
    mask = mask.to('cpu')
    num, h, w = mask.shape
    #bin_mask = np.zeros((h, w))

    # # convert to numpy and reverse the dimensions
    # mask = mask.numpy()
    # mask = np.transpose(mask, (1, 2, 0))
    # mask = mask.astype(np.uint8)
    # mask = mask * 255
    # mask = mask.squeeze()
    # print(mask.shape)
    # print(mask)

    mask = mask.type(torch.float32)

    for i, score in enumerate(scores):
        mask[i] = mask[i] * score

    bin_mask = torch.sum(mask, dim=0)

    # reduce value above 1 to one in bin_mask
    bin_mask[bin_mask > 1] = 1
    print(bin_mask.max())
    bin_mask = (bin_mask + .5) * 170
    bin_mask = np.expand_dims(bin_mask, axis=2)

    out_img = np.concatenate([img, bin_mask], axis=2)

    #print(out_img.shape)

    # print(bin_mask.shape)
    filename = os.path.join(output_dir, os.path.basename(img_fn))
    cv2.imwrite(filename, out_img)

    # plt.figure(figsize=(200, 200))
    # title = ["Predicted", "Ground Truth"]
    # display_list = [out1, out2]
    # for j in range(len(display_list)):
    #     plt.subplot(1, len(display_list), j + 1)
    #     plt.title(title[j])
    #     type(display_list[j])
    #     plt.imshow(display_list[j].get_image()[:, :, ::-1])
    #     plt.axis("off")

    # plt.savefig(os.path.join(cfg.OUTPUT_DIR,
    #             f"prediction-{os.path.basename(image['file_name'])}"))
    # plt.close()
