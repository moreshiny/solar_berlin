from model import Model


train_path = "data/curated_1/train_curated_1_final/"
test_path = "data/curated_1/test_curated_1_final/"

layer_names = [
    "block_1_expand_relu",   # 64x64
    "block_3_expand_relu",   # 32x32
    "block_6_expand_relu",   # 16x16
    "block_13_expand_relu",  # 8x8
    "block_16_project",      # 4x4
]

model = Model(
    train_path,
    test_path,
    layer_names,
    epochs=5,
    batch_size=16,
)

model.model_history()
