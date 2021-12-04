import os
<<<<<<< HEAD
import random
=======
>>>>>>> 4c82acd (First working version of data selector with multiclass)
import shutil
import glob
from PIL import Image
from osgeo import gdal
from osgeo import ogr

<<<<<<< HEAD
from selection.errors import InvalidPathError, AbsolutePathError, OutputPathExistsError
from selection.errors import InvalidTileSizeError, InsuffientDataError
=======
from extraction.extraction import select_random_map_images, copy_image_files
>>>>>>> 4c82acd (First working version of data selector with multiclass)

RASTER_TILE_SIZE = 10_000


<<<<<<< HEAD
class DataHandler():
    def __init__(self):
        pass

    @staticmethod
    def _verify_path(path_str):
        if not os.path.exists(path_str):
            raise InvalidPathError(f"Path {path_str} does not exist")
        if os.path.isabs(path_str):
            raise AbsolutePathError(f"Path {path_str} is absolute")


class DataExtractor(DataHandler):
    def __init__(self, input_path):
        self._verify_path(input_path)


class DataSelector(DataHandler):

    def __init__(self, input_path, testing: bool = False):
        self.extractor = DataExtractor(input_path)
        self.input_path = input_path
        self._testing = testing

    def select_data(self, tile_size, train_n, test_n, output_path, random_seed=0, lossy=False):

        # self.extractor.extract(tile_size=tile_size)

        self._verify_path(output_path)
        # if os.path.isabs(output_path):
        #     raise AbsolutePathError(f"Output path {output_path} is absolute")
=======
class DataSelector():

    def __init__(self, input_path, testing: bool = False):

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path {input_path} does not exist")

        if os.path.isabs(input_path):
            raise ValueError(f"Input path {input_path} is absolute")

        self.input_path = input_path
        self.output_path = None
        self._testing = testing

    def select_data(self, tile_size, train_n, test_n, output_path, random_seed=0, multiclass=True):

        if os.path.isabs(output_path):
            raise ValueError(f"Output path {output_path} is absolute")
>>>>>>> 4c82acd (First working version of data selector with multiclass)

        self.output_path = os.path.join(
            output_path, f"selected_tiles_{tile_size}_{train_n}_{test_n}_{random_seed}")

        if not type(tile_size) is int or tile_size < 1:
<<<<<<< HEAD
            raise InvalidTileSizeError(
                f"Tile size {tile_size} is not an integer or less than 1")

        if RASTER_TILE_SIZE % tile_size != 0:
            if lossy:
                self.raster_tile_size = RASTER_TILE_SIZE // tile_size * tile_size
            else:
                raise InvalidTileSizeError(
                    f"tile_size must be a factor of {RASTER_TILE_SIZE} or raster edges will be discarded. Set lossy=True to allow this.")
        else:
            self.raster_tile_size = RASTER_TILE_SIZE
=======
            raise ValueError(
                f"Tile size {tile_size} is not an integer or less than 1")

        if RASTER_TILE_SIZE % tile_size != 0:
            raise ValueError(
                f"tile_size must be a factor of {RASTER_TILE_SIZE}")
>>>>>>> 4c82acd (First working version of data selector with multiclass)

        current_tile_path = os.path.join(
            self.input_path, "tiled_" + str(tile_size))

        if self._request_size_too_large(tile_size, train_n, test_n):
<<<<<<< HEAD
            raise InsuffientDataError(
=======
            raise ValueError(
>>>>>>> 4c82acd (First working version of data selector with multiclass)
                f"Not enough tiles to extract {train_n} training and {test_n} testing tiles")

        if os.path.exists(current_tile_path):
            print(
                f"Tile size {tile_size} already extracted, proceeding to selection")
        else:
            self._extract_data(tile_size, current_tile_path, train_n, test_n)

<<<<<<< HEAD
        file_lists = self._select_random_map_images(
=======
        file_lists = select_random_map_images(
>>>>>>> 4c82acd (First working version of data selector with multiclass)
            train_size=train_n,
            test_size=test_n,
            input_path=current_tile_path,
            random_seed=random_seed,
        )

<<<<<<< HEAD
        self._copy_image_files(
=======
        copy_image_files(
>>>>>>> 4c82acd (First working version of data selector with multiclass)
            file_lists,
            input_path=current_tile_path,
            output_path=self.output_path,
        )

    def _request_size_too_large(self, tile_size, train_n, test_n):
<<<<<<< HEAD
        if self._testing:
            # limit input to 32 tiles for faster testing
            raster_tile_size = min(self.raster_tile_size, tile_size*4)
        else:
            raster_tile_size = self.raster_tile_size
=======

        if self._testing:
            # limit input to 32 tiles for faster testing
            raster_tile_size = min(RASTER_TILE_SIZE, tile_size*4)
        else:
            raster_tile_size = RASTER_TILE_SIZE
>>>>>>> 4c82acd (First working version of data selector with multiclass)

        raster_map_fns = glob.glob(os.path.join(
            self.input_path, "raster", "*.tif"))

        total_no_of_pixels = len(raster_map_fns) * raster_tile_size**2
        requested_no_of_pixels = (train_n + test_n) * tile_size**2

        return total_no_of_pixels < requested_no_of_pixels

<<<<<<< HEAD
    @staticmethod
    def _select_random_map_images(train_size: int, test_size: int,
                                  input_path: str, random_seed: int = 0) -> list:
        """Selects a random subset of map images from the input path and returns
        them as a List of two Lists (one each for train and test) of tuples pairs
        of map and mask images.

        Args:
            train_size (int): number of training images to select
            test_size (int): number of test images to select
            input_path (str): location of input files

        Returns:
            list: List of lists of tuples pairs of map and mask images
        """
        # get all files in input directory
        files = os.listdir(input_path)
        files_map = [file for file in files if "map" in file]
        files_mask = [
            file for file in files if "msk" in file or "mask" in file]

        # sort files by name
        files_map.sort()
        files_mask.sort()

        files_zipped = list(zip(files_map, files_mask))

        # shuffle the file pairs
        random.seed(random_seed)
        random.shuffle(files_zipped)

        # select train and test from (shuffled) front
        files_train = files_zipped[:train_size]
        files_test = files_zipped[train_size:train_size+test_size]

        # for now just return these as a list
        # TODO define clearer file type for loaded data
        return [files_train, files_test]

    @staticmethod
    def _copy_image_files(image_files: list, input_path: str,
                          output_path: str, delete_existing_output_path_no_warning=False):
        """Copy image files from the input path to the output path.

        Args:
            image_files (list): Filenames as returned by select_random_map_images
            input_path (str): original file location
            output_path (str): file location to copy to
            delete_existing_output_path_no_warning (bool, optional): Delete output
            path first if it exists, without warning. Defaults to False.
        """

        # if we have been asked to, delete existing out path without warning
        # TODO is this safe?
        if delete_existing_output_path_no_warning and os.path.exists(output_path):
            shutil.rmtree(output_path)

        # create output path if it doesn't exist or end if it does
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        else:
            raise OutputPathExistsError("At least one of the output directory already exists."
                                        "\nSet delete_existing=True to remove it.")

        # get file names into a dict for easier processing
        files = {}
        files["train"] = image_files[0]
        files["test"] = image_files[1]

        for subfolder in ["train", "test"]:
            output_path_subfolder = os.path.join(output_path, subfolder)

            # create output folder if it doesn't exist
            if not os.path.exists(output_path_subfolder):
                os.makedirs(output_path_subfolder)

            # copy files to output folder
            for file_tuple in files[subfolder]:
                for file_path in file_tuple:
                    full_path_in = os.path.join(input_path, file_path)
                    full_path_out = os.path.join(
                        output_path_subfolder, file_path)
                    shutil.copy(full_path_in, full_path_out)

    def _extract_data(self, tile_size, current_tile_path, train_n, test_n):
=======
    def _extract_data(self, tile_size, current_tile_path, train_n, test_n):

>>>>>>> 4c82acd (First working version of data selector with multiclass)
        if self._testing:
            raster_tile_size = min(RASTER_TILE_SIZE, tile_size*4)
        else:
            raster_tile_size = RASTER_TILE_SIZE

        raster_map_fns = glob.glob(os.path.join(
            self.input_path, "raster", "*.tif"))

        if not os.path.exists(current_tile_path):
            os.makedirs(current_tile_path)

        for raster_map_fn in raster_map_fns:
            map_tile_name = os.path.basename(raster_map_fn)[0:-4]
            vector_fn = os.path.join(
                self.input_path, "vector", map_tile_name, map_tile_name + ".shp")

            vector_file = ogr.Open(vector_fn)
            vector_layer = vector_file.GetLayer()
            x_min, x_max, y_min, y_max = vector_layer.GetExtent()

            pixel_size = .2

            temp_path = os.path.join(current_tile_path, "temp")

            if not os.path.exists(temp_path):
                os.makedirs(temp_path)

            raster_msk_fn = os.path.join(
                temp_path, map_tile_name + "_msk.tif")

            x_res = int((x_max - x_min) / pixel_size)
            y_res = int((y_max - y_min) / pixel_size)
            raster_msk_file = gdal.GetDriverByName('GTiff').Create(
                raster_msk_fn, x_res, y_res, 1, gdal.GDT_Int16)

            raster_msk_file.SetGeoTransform(
                (x_min, pixel_size, 0, y_max, 0, -pixel_size))

            # get the first raster band fill the raster band with -1s
            raster_msk_band = raster_msk_file.GetRasterBand(1)
            raster_msk_band.SetNoDataValue(-1)

            # overwrite the band with each polygon's 'eig_kl_pv' (0, 1, 2 or 3)
            # any area not covered by a vector object remains at -1 ("no roof")
            gdal.RasterizeLayer(raster_msk_file, [1], vector_layer,
                                options=["ATTRIBUTE=eig_kl_pv"])

            raster_map_file = gdal.Open(raster_map_fn)

            geoTransform = raster_map_file.GetGeoTransform()
            x_min_s = geoTransform[0]
            y_max_s = geoTransform[3]

            # convert extent to pixel coordinates
            x_min_p = int((x_min_s - x_min) / pixel_size)
            y_min_p = int((y_max - y_max_s) / pixel_size)

            x_max_p = x_min_p + raster_tile_size
            y_max_p = y_min_p + raster_tile_size

            print(x_min_p, x_max_p, y_min_p, y_max_p)

            # set to none so gdal actually writes the data
            raster_map_file = None
            raster_msk_file = None

            raster_msk_clip_fn = os.path.join(
                temp_path, map_tile_name + "_msk_clip.tif")

            # crop the raster to the extent of the vector
            gdal.Translate(raster_msk_clip_fn, raster_msk_fn, srcWin=[
                x_min_p,
                y_min_p,
                raster_tile_size,
                raster_tile_size,
            ])

            raster_map_file = gdal.Open(raster_map_fn)
            raster_map_array = raster_map_file.ReadAsArray()

            raster_msk_file = gdal.Open(raster_msk_clip_fn)
            raster_mask_array = raster_msk_file.ReadAsArray()

            # shift categories by 1 to make 0 the lowest category
            # 0 is now "no roof", 1, 2, 3, 4 are the pv categories
            raster_mask_array = raster_mask_array + 1
            # shift categories into visible range, 4 is the max possible value
            # 0 is now "no roof", 63, 127, 191, 255 are the pv categories
            raster_mask_array = raster_mask_array / 4 * 255

            for x_coord in range(0, raster_tile_size, tile_size):
                for y_coord in range(0, raster_tile_size, tile_size):
                    sub_array_mask = raster_mask_array[
                        x_coord: (x_coord + tile_size),
                        y_coord: (y_coord + tile_size),
                    ]

                    im = Image.fromarray(sub_array_mask).convert("L")

                    im.save(os.path.join(
                        current_tile_path, f"{map_tile_name}_{x_coord}_{y_coord}_msk.png"
                    ))

                    sub_array_map = raster_map_array[
                        :,
                        x_coord:x_coord + tile_size,
                        y_coord:y_coord + tile_size,
                    ]

                    sub_array_map = sub_array_map.transpose(1, 2, 0)

                    im = Image.fromarray(sub_array_map).convert("RGB")
                    im.save(os.path.join(
                        current_tile_path, f"{map_tile_name}_{x_coord}_{y_coord}_map.png"
                    ))

            raster_map_fn = None
            raster_msk_fn = None

            if not self._testing:
                shutil.rmtree(temp_path)

            if self._testing:
                break
