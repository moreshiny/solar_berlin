import os
import random
import shutil
import glob
from PIL import Image
from osgeo import gdal
from osgeo import ogr

from selection.errors import InvalidPathError, AbsolutePathError, OutputPathExistsError
from selection.errors import InvalidTileSizeError, InsuffientDataError

RASTER_TILE_SIZE = 10_000


class DataHandler():
    def __init__(self):
        pass

    @classmethod
    def _verify_input_path(cls, path_str):
        cls._verify_any_path(path_str, pathname="Input path")
        if not os.path.exists(path_str):
            raise InvalidPathError(f"Input path {path_str} does not exist")

    @classmethod
    def _verify_output_path(cls, path_str):
        cls._verify_any_path(path_str, pathname="Output path")
        if os.path.exists(path_str):
            raise OutputPathExistsError(
                f"Output path {path_str} already exists")

    @staticmethod
    def _verify_any_path(path_str, pathname="Path"):
        if os.path.isabs(path_str):
            raise AbsolutePathError(f"{pathname} {path_str} is absolute")
        if path_str == "":
            raise InvalidPathError(f"{pathname} {path_str} is empty")


class DataExtractor(DataHandler):
    def __init__(self, input_path, output_path, tile_size, lossy=False, testing=False):
        self._testing = testing
        self.tile_size = tile_size

        self._verify_input_path(input_path)
        self._input_path = input_path
        self._input_raster_fns = glob.glob(
            os.path.join(self._input_path, "raster", "*.tif")
        )

        if not type(self.tile_size) is int or self.tile_size < 1:
            raise InvalidTileSizeError(
                f"Tile size {self.tile_size} is not an integer or less than 1")

        if RASTER_TILE_SIZE % self.tile_size != 0:
            if lossy:
                self.raster_tile_size = RASTER_TILE_SIZE // self.tile_size * self.tile_size
            else:
                raise InvalidTileSizeError(
                    f"tile_size must be a factor of {RASTER_TILE_SIZE} or raster edges will be discarded. Set lossy=True to allow this.")
        else:
            self.raster_tile_size = RASTER_TILE_SIZE

        if self._testing:
            print(f"Testing mode: {self._testing}")
            # limit to 16 (4*4) tiles for testing
            self.raster_tile_size = min(
                self.raster_tile_size, self.tile_size*4)
        else:
            self.raster_tile_size = self.raster_tile_size

        self.tile_path = os.path.join(output_path, f"tiles_{self.tile_size}")
        try:
            self._verify_output_path(self.tile_path)
        except OutputPathExistsError:
            output_map_tile_fns = glob.glob(
                os.path.join(self.tile_path, "*_map.png"))
            output_msk_tile_fns = glob.glob(
                os.path.join(self.tile_path, "*_msk.png"))

            expected_tile_nos = len(self._input_raster_fns) * \
                (self.raster_tile_size**2 // self.tile_size**2)

            if len(output_map_tile_fns) != expected_tile_nos or len(output_msk_tile_fns) != expected_tile_nos:
                raise OutputPathExistsError(
                    f"Output path {self.tile_path} exists and does not contain the expected number of tiles")
            else:
                self.total_tiles = len(output_map_tile_fns)
        else:
            self.total_tiles = 0
            self._extract_data(self.tile_size, self.tile_path)

    def _extract_data(self, tile_size, tile_path):
        if not os.path.exists(tile_path):
            os.makedirs(tile_path)

        for raster_map_fn in self._input_raster_fns:
            # create a temporary working directory
            temp_path = os.path.join(tile_path, "temp")
            self._verify_output_path(temp_path)
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)

            # get the base tile name we are working on
            map_tile_name = os.path.basename(raster_map_fn)[0:-4]

            # open the vector (mask) file to get the extent
            vector_fn = os.path.join(
                self._input_path,
                "vector",
                map_tile_name,
                map_tile_name + ".shp",
            )
            vector_file = ogr.Open(vector_fn)
            vector_layer = vector_file.GetLayer()
            x_min, x_max, y_min, y_max = vector_layer.GetExtent()

            # define a temporary rastr for the mask
            tmp_msk_raster_fn = os.path.join(
                temp_path,
                map_tile_name + "_msk.tif"
            )
            # set pixel_size to match the map raster
            pixel_size = .2
            # resolution based on the extent we got from the vector
            x_res = int((x_max - x_min) / pixel_size)
            y_res = int((y_max - y_min) / pixel_size)
            gtiff_driver = gdal.GetDriverByName("GTiff")
            tmp_msk_raster = gtiff_driver.Create(
                tmp_msk_raster_fn,
                x_res,
                y_res,
                1,  # one band
                gdal.GDT_Int16
            )
            # GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
            # GT(1) w-e pixel resolution / pixel width.
            # GT(2) row rotation (typically zero).
            # GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
            # GT(4) column rotation (typically zero).
            # GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).
            tmp_msk_raster.SetGeoTransform(
                (x_min, pixel_size, 0, y_max, 0, -pixel_size)
            )

            # set the first (and only) band default value to -1
            tmp_msk_raster.GetRasterBand(1).SetNoDataValue(-1)

            # overwrite the band with each polygon's 'eig_kl_pv' (0, 1, 2 or 3)
            # any area not covered by a vector object remains at -1 ("no roof")
            gdal.RasterizeLayer(
                tmp_msk_raster,
                [1],  # band one (the only band)
                vector_layer,
                options=["ATTRIBUTE=eig_kl_pv"]
            )
            # ensure the raster is closed so that changes are written to disk
            del tmp_msk_raster

            # open the map raster to get the extent
            raster_map_file = gdal.Open(raster_map_fn)
            geoTransform = raster_map_file.GetGeoTransform()
            x_min_s = geoTransform[0]
            y_max_s = geoTransform[3]
            # convert extent to pixel coordinates
            x_min_p = int((x_min_s - x_min) / pixel_size)
            y_min_p = int((y_max - y_max_s) / pixel_size)

            # create a temporary raster for the clipped mask
            tmp_msk_raster_clip_fn = os.path.join(
                temp_path, map_tile_name + "_msk_clip.tif"
            )
            # crop mask rater to the extent of the map raster
            # this cuts any objects overlapping the edges at the edge of the map
            gdal.Translate(tmp_msk_raster_clip_fn, tmp_msk_raster_fn, srcWin=[
                x_min_p,
                y_min_p,
                self.raster_tile_size,
                self.raster_tile_size,
            ])

            # load the map and tempoary mask as numpy arrays for processing
            raster_map_array = gdal.Open(raster_map_fn).ReadAsArray()
            tmp_msk_raster_clip_array = gdal.Open(
                tmp_msk_raster_clip_fn).ReadAsArray()

            # shift categories by 1 to make 0 the lowest category
            # 0 is now "no roof", 1, 2, 3, 4 are the pv categories
            tmp_msk_raster_clip_array = tmp_msk_raster_clip_array + 1
            # shift categories into visible range, 4 is the max possible value
            # 0 is now "no roof", 63, 127, 191, 255 are the pv categories
            tmp_msk_raster_clip_array = tmp_msk_raster_clip_array / 4 * 255

            # shift a window of tile_size over the rasters and save the tiles
            # as png images - if self.raster_tile_size is smaller than the
            # raster extent, a part of the map is omited
            for x_coord in range(0, self.raster_tile_size, tile_size):
                for y_coord in range(0, self.raster_tile_size, tile_size):
                    sub_array_msk = tmp_msk_raster_clip_array[
                        x_coord: (x_coord + tile_size),
                        y_coord: (y_coord + tile_size),
                    ]
                    # same msk tiles as greyscale
                    im = Image.fromarray(sub_array_msk).convert("L")
                    im.save(os.path.join(
                        tile_path, f"{map_tile_name}_{x_coord}_{y_coord}_msk.png"
                    ))

                    sub_array_map = raster_map_array[
                        :,
                        x_coord:x_coord + tile_size,
                        y_coord:y_coord + tile_size,
                    ]
                    # save map tiles as channel-last RGB
                    sub_array_map = sub_array_map.transpose(1, 2, 0)
                    im = Image.fromarray(sub_array_map).convert("RGB")
                    im.save(os.path.join(
                        tile_path, f"{map_tile_name}_{x_coord}_{y_coord}_map.png"
                    ))

                    # keep track of the number of tiles created
                    self.total_tiles += 1

            # when testing, keep temporary files and stop after the first map tile
            if not self._testing:
                shutil.rmtree(temp_path)
            if self._testing:
                break


class DataSelector(DataHandler):

    def __init__(
            self,
            extractor,
            output_path,
            train_n,
            test_n,
            random_seed=0,
            testing: bool = False):

        self._testing = testing

        self.extractor = extractor

        self._verify_request_size(train_n, test_n)
        self.train_n = train_n
        self.test_n = test_n

        self._verify_superdirectory_path(output_path)
        output_subdir = self._subdir_name(
            extractor.tile_size,
            train_n,
            test_n,
            random_seed
        )
        full_output_path = os.path.join(output_path, output_subdir)
        self._verify_output_path(full_output_path)
        self.output_path = full_output_path

        file_lists = self._select_random_map_images(
            train_size=train_n,
            test_size=test_n,
            input_path=self.extractor.tile_path,
            random_seed=random_seed,
        )

        self._copy_image_files(
            file_lists,
            input_path=self.extractor.tile_path,
            output_path=self.output_path,
        )

    def _verify_request_size(self, train_n, test_n):
        if train_n + test_n > self.extractor.total_tiles:
            raise InsuffientDataError(
                f"Requested {train_n} training tiles and {test_n} testing tiles, but only {self.extractor.total_tiles} tiles available.")

    @classmethod
    def _verify_superdirectory_path(cls, output_path):
        try:
            cls._verify_output_path(output_path)
        except OutputPathExistsError:
            pass  # ignore that output path exists
        # instead ensure that the output path exists so that we can create subfolders
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    @staticmethod
    def _subdir_name(tile_size, train_n, test_n, random_seed):
        return f"selected_tiles_{tile_size}_{train_n}_{test_n}_{random_seed}"

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
