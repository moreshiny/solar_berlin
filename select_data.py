import os

from selection.selection import DataExtractor
from selection.selection import DataSelector

CONVERTED_DATA_PATH = os.path.join("data", "converted")
EXTRACTED_PATH = os.path.join("data", "extracted")
SELECTED_PATH = os.path.join("data", "selected")
TILE_SIZE = 512

TRAIN_N = 100
TEST_N = 20

RANDOM_SEED = 42

extractor = DataExtractor(
    input_path=CONVERTED_DATA_PATH,
    output_path=EXTRACTED_PATH,
    tile_size=TILE_SIZE,
    lossy=True,
)

selector = DataSelector(
    extractor=extractor,
    output_path=SELECTED_PATH,
    train_n=TRAIN_N,
    test_n=TEST_N,
    random_seed=RANDOM_SEED,
)

print(selector.output_path, selector.train_n, selector.test_n)
