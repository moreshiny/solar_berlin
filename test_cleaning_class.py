from class_automated_data_cleaning import DataCleaning
from class_unet_resnet101v2 import Unet




output_classes = 5  # number of categorical classes.
input_shape = (512, 512, 3)  # input size

#accuracy threshild for the cleaning

# calling the model.
model = Unet(
    output_classes=output_classes,
    input_shape=input_shape,
    drop_out=False,
    drop_out_rate={"512": 0.275, "256": 0.3, "128": 0.325, "64": 0.35},
    multiclass=bool(output_classes - 1),
)


path_checkpoint = "logs/12_09_2021_22_25_55/checkpoint.ckpt"

model.load_weights(path_checkpoint)

path_to_clean = "data/selected_tiles_512_4000_1000_42_cleaned/train"

cleaning = DataCleaning(
        path_to_clean = path_to_clean,
        input_shape = (512, 512, 3),
        model = model,
)

cleaning.cleaning(proportion=0.05)


print("Done")



path_to_clean = "data/selected_tiles_512_4000_1000_42_cleaned/test"

cleaning = DataCleaning(
        path_to_clean = path_to_clean,
        input_shape = (512, 512, 3),
        model = model,
)

cleaning.cleaning(proportion=0.05)


print("Done")



