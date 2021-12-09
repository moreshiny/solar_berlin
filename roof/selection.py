import os
import random
import shutil
import glob
from PIL import Image
from osgeo import gdal
from osgeo import ogr
from abc import ABC
<<<<<<< HEAD
import json
import geopandas as gpd
from shapely.geometry import Polygon, box
=======
>>>>>>> bdb2ba5 (Combine classes into a single roof module)

from roof.errors import (
    AbsolutePathError,
    InsuffientDataError,
    InvalidPathError,
    InvalidTileSizeError,
    OutputPathExistsError,
)

# input raster tiles are all 10k x 10k pixels
RASTER_TILE_SIZE = 10_000


class DataHandler(ABC):
    """
    Abstract class as parent for data handling classes.
    """

    @classmethod
    def _verify_input_path(cls, path_str: str) -> None:
        """ Standard verification of input paths.

        Args:
            path_str (str): A path-like to be checked.

        Raises:
            InvalidPathError: If path does not exist.
        """
        cls._verify_any_path(path_str, pathname="Input path")
        if not os.path.exists(path_str):
            raise InvalidPathError(f"Input path {path_str} does not exist")

    @classmethod
    def _verify_output_path(cls, path_str: str) -> None:
        """ Standard verification of output paths.

        Args:
            path_str (str): A path-like to be checked.

        Raises:
            OutputPathExistsError: If path already exists.
        """
        cls._verify_any_path(path_str, pathname="Output path")
        if os.path.exists(path_str):
            raise OutputPathExistsError(
                f"Output path {path_str} already exists")

    @staticmethod
    def _verify_any_path(path_str: str, pathname: str = "Path"):
        """ Standard verification applicable to any path.

        Args:
            path_str (str): A path-like to be checked.
            pathname (str, optional): Description of path (used in error
                messages). Defaults to "Path".

        Raises:
            AbsolutePathError: If path is absolute.
            InvalidPathError: If path is empty or otherwise invalid.
        """
        if os.path.isabs(path_str):
            raise AbsolutePathError(f"{pathname} {path_str} is absolute")
        if path_str == "":
            raise InvalidPathError(f"{pathname} {path_str} is empty")


class DataExtractor(DataHandler):
    """
    DataExtractor takes matching raster and vector tiles and extracts them into
    usable map-mask pairs for furhter processing.
    """

    def __init__(self, input_path: str, output_path: str, tile_size: int, lossy=False, testing=False):
        """ Initialize DataExtractor.

        Args:
            input_path (str): A path-like to the input data, which should be a
                directory containing rater tiles in a "raster" subdirectory and
                matching vector tiles in a "vector" subdirectory. Raster tiles
                are assumed to be 10k x 10k pixels.
            output_path (str): A path-like to the output directory. A subdirectory
                will be created for the output tiles within this directory.
            tile_size (int): The size of the output tiles in pixels. Extracted
                tiles are always square of tile_size x tile_size pixels.
            lossy (bool, optional): Whether to permit tile_size values that
                result in data loss at the edge of a raster. When False, only
                factors of 10k are permitted as tile_size. Defaults to False.
            testing (bool, optional): When true, only a small sub-portion of
                the first raster encountered will be extracted. Defaults to False.

        Raises:
            InvalidTileSizeError: If tile_size is not a valid integer (or a
                factor of 10k when Lossy is False).
            OutputPathExistsError: If the output path already exists and does
                not already contain matching extracted tiles.
        """
        self._testing = testing
        self.tile_size = tile_size
<<<<<<< HEAD
        self.total_tiles = 0

        self._verify_input_path(input_path)
        self._input_path = input_path
        self._input_raster_fns = sorted(glob.glob(
            os.path.join(self._input_path, "raster", "*.tif")
        ))
=======

        self._verify_input_path(input_path)
        self._input_path = input_path
        self._input_raster_fns = glob.glob(
            os.path.join(self._input_path, "raster", "*.tif")
        )
>>>>>>> bdb2ba5 (Combine classes into a single roof module)

        if not type(self.tile_size) is int or self.tile_size < 1:
            raise InvalidTileSizeError(
                f"Tile size {self.tile_size} is not an integer or less than 1")

        if RASTER_TILE_SIZE % self.tile_size != 0:
            if lossy:
                self.raster_tile_size =\
                    RASTER_TILE_SIZE // self.tile_size * self.tile_size
            else:
                raise InvalidTileSizeError(
                    f"""tile_size must be a factor of {RASTER_TILE_SIZE} or raster
                        edges will be discarded. Set lossy=True to allow this."""
                )
        else:
            self.raster_tile_size = RASTER_TILE_SIZE

        if self._testing:
            print(f"Testing mode: {self._testing}")
            # limit to 16 (4*4) tiles for faster testing
            self.raster_tile_size = min(
                self.raster_tile_size, self.tile_size*4)
        else:
            self.raster_tile_size = self.raster_tile_size

        self.tile_path = os.path.join(output_path, f"tiles_{self.tile_size}")
        try:
            self._verify_output_path(self.tile_path)
        except OutputPathExistsError:
            output_map_tile_fns = glob.glob(
<<<<<<< HEAD
                os.path.join(self.tile_path, "**", "*_map.png"),
                recursive=True,
            )
            output_msk_tile_fns = glob.glob(
                os.path.join(self.tile_path, "**", "*_msk.png"),
                recursive=True,
            )
=======
                os.path.join(self.tile_path, "*_map.png"))
            output_msk_tile_fns = glob.glob(
                os.path.join(self.tile_path, "*_msk.png"))
>>>>>>> bdb2ba5 (Combine classes into a single roof module)

            expected_tile_nos = len(self._input_raster_fns) * \
                (self.raster_tile_size**2 // self.tile_size**2)

            if len(output_map_tile_fns) != expected_tile_nos\
                    or len(output_msk_tile_fns) != expected_tile_nos:
<<<<<<< HEAD
                self.total_tiles = 0
                self._extract_data(self.tile_size, self.tile_path)
            else:
                print("Tiles already extracted. Checking coco.")
                coco_fns = glob.glob(
                    os.path.join(self.tile_path, "**", "*.json")
                )
                expected_coco_nos = len(self._input_raster_fns)
                if not len(coco_fns) == expected_coco_nos:
                    print("Coco jsons incomplete. Recreating.")
                    self.total_tiles = 0
                    self._extract_data(
                        self.tile_size,
                        self.tile_path,
                        coco_only=True,
                    )
                else:
                    print("Coco json present. Extraction complete")
                    self.total_tiles = len(output_map_tile_fns)
=======
                raise OutputPathExistsError(
                    f"""Output path {self.tile_path} exists and does not contain
                        the expected number of tiles"""
                )
            else:
                self.total_tiles = len(output_map_tile_fns)
>>>>>>> bdb2ba5 (Combine classes into a single roof module)
        else:
            self.total_tiles = 0
            self._extract_data(self.tile_size, self.tile_path)

<<<<<<< HEAD
    def _extract_data(self, tile_size: int, tile_path: str, coco_only: bool = False):
=======
    def _extract_data(self, tile_size: int, tile_path: str):
>>>>>>> bdb2ba5 (Combine classes into a single roof module)
        """ Main method for extracting data. For each raster tile encountered,
            the corresponding vector file is rasterised, clipped to the size of
            the raster and converted to a mask image. Creates a temporary
            directory in the output path to hold intermediate states.

        Args:
            tile_size (int): Size of the tiles to be extracted.
            tile_path (str): The subdirectory in which to save the tiles.
<<<<<<< HEAD
            coco_only (bool, optional): Whether to only extract coco jsons (if)
                tiles have already been extracted. Defaults to False.
=======
>>>>>>> bdb2ba5 (Combine classes into a single roof module)
        """
        if not os.path.exists(tile_path):
            os.makedirs(tile_path)

<<<<<<< HEAD
        for count, raster_map_fn in enumerate(self._input_raster_fns):
            print(
                f"Processing #{count+1} of {len(self._input_raster_fns)}: {raster_map_fn}..."
            )
            tile_coco_only = coco_only

            subfolder_fn = os.path.basename(raster_map_fn).split(".")[0]
            if not tile_coco_only:
                if os.path.exists(os.path.join(tile_path, subfolder_fn)):
                    found_tile_nos = len(glob.glob(
                        os.path.join(tile_path, subfolder_fn, "*_map.png")))
                    expected_tiles = (self.raster_tile_size**2 // tile_size**2)
                    if found_tile_nos == expected_tiles:
                        print(
                            f"{subfolder_fn} already extracted. Checking coco...")
                        json_path = os.path.join(
                            tile_path,
                            subfolder_fn,
                            subfolder_fn + ".json",
                        )
                        if os.path.exists(json_path):
                            print(f"{subfolder_fn} coco present. Skipping...")
                            self.total_tiles += expected_tiles
                            continue
                        else:
                            print(
                                f"{subfolder_fn} coco not found. Generating...")
                            tile_coco_only = True
                    else:
                        # check whether a vector file exists for this raster
                        # if not, skip this raster
                        shp_path = os.path.join(
                            self._input_path,
                            "vector",
                            subfolder_fn,
                            subfolder_fn + ".shp",
                        )
                        if not os.path.exists(shp_path):
                            print(
                                f"No vector file for {subfolder_fn}. Skipping...")
                            continue
                        else:
                            raise OutputPathExistsError(
                                f"""Output path {tile_path}/{subfolder_fn} exists and does not contain
                                the expected number of tiles"""
                            )

            # create a temporary working directory
            temp_path = os.path.join(tile_path, subfolder_fn, "temp")
=======
        for raster_map_fn in self._input_raster_fns:
            print(f"Processing {raster_map_fn}...")
            # create a temporary working directory
            temp_path = os.path.join(tile_path, "temp")
>>>>>>> bdb2ba5 (Combine classes into a single roof module)
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
            try:
                vector_layer = vector_file.GetLayer()
            except AttributeError:
                print(
                    f"No vector objects for present for {map_tile_name}, skipping"
                )
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path)
                continue

            x_min, x_max, y_min, y_max = vector_layer.GetExtent()

<<<<<<< HEAD
            shapefile = gpd.read_file(vector_fn)
            # set pixel_size to match the map raster
            pixel_size = .2

            if not tile_coco_only:
                # define a temporary rastr for the mask
                tmp_msk_raster_fn = os.path.join(
                    temp_path,
                    map_tile_name + "_msk.tif"
                )

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
=======
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
>>>>>>> bdb2ba5 (Combine classes into a single roof module)

            # open the map raster to get the extent
            raster_map_file = gdal.Open(raster_map_fn)
            geoTransform = raster_map_file.GetGeoTransform()
            x_min_s = geoTransform[0]
            y_max_s = geoTransform[3]
            # convert extent to pixel coordinates
            x_min_p = int((x_min_s - x_min) / pixel_size)
            y_min_p = int((y_max - y_max_s) / pixel_size)

<<<<<<< HEAD
            if not tile_coco_only:
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
                tmp_msk_raster_clip = gdal.Open(tmp_msk_raster_clip_fn)

                tmp_msk_raster_clip_array = tmp_msk_raster_clip.ReadAsArray()

                # shift categories by 1 to make 0 the lowest category
                # 0 is now "no roof", 1, 2, 3, 4 are the pv categories
                tmp_msk_raster_clip_array = tmp_msk_raster_clip_array + 1
                # shift categories into visible range, 4 is the max possible value
                # 0 is now "no roof", 63, 127, 191, 255 are the pv categories
                tmp_msk_raster_clip_array = tmp_msk_raster_clip_array / 4 * 255

            tmp_xmin = raster_map_file.GetGeoTransform()[0]
            tmp_ymax = raster_map_file.GetGeoTransform()[3]

            # create a coco json file for this tile
            coco_json = {
                "info": {
                    "description": "",
                    "url": "",
                    "version": "",
                    "year": 2021,
                    "contributor": "",
                    "date_created": "",
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [
                    {
                        "id": 0,
                        "name": "pv1",
                        "supercategory": "roof",
                    },
                    {
                        "id": 1,
                        "name": "pv2",
                        "supercategory": "roof",
                    },
                    {
                        "id": 2,
                        "name": "pv3",
                        "supercategory": "roof",
                    },
                    {
                        "id": 3,
                        "name": "pv4",
                        "supercategory": "roof",
                    },
                ],
            }
=======
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
>>>>>>> bdb2ba5 (Combine classes into a single roof module)

            # shift a window of tile_size over the rasters and save the tiles
            # as png images - if self.raster_tile_size is smaller than the
            # raster extent, a part of the map is omited
<<<<<<< HEAD
            for y_coord in range(0, self.raster_tile_size, tile_size):
                for x_coord in range(0, self.raster_tile_size, tile_size):

                    filename = f"{map_tile_name}_{y_coord}_{x_coord}"
                    if not tile_coco_only:
                        sub_array_msk = tmp_msk_raster_clip_array[
                            y_coord: (y_coord + tile_size),
                            x_coord: (x_coord + tile_size),
                        ]
                        # same msk tiles as greyscale
                        im = Image.fromarray(sub_array_msk).convert("L")
                        im.save(os.path.join(
                            tile_path, subfolder_fn, f"{filename}_msk.png"
                        ))

                        sub_array_map = raster_map_array[
                            :,
                            y_coord:y_coord + tile_size,
                            x_coord:x_coord + tile_size,
                        ]
                        # save map tiles as channel-last RGB
                        sub_array_map = sub_array_map.transpose(1, 2, 0)
                        im = Image.fromarray(sub_array_map).convert("RGB")
                        im.save(os.path.join(
                            tile_path, subfolder_fn, f"{filename}_map.png"
                        ))

                    image_id = filename
                    # append the image info to the coco json
                    coco_json["images"].append({
                        "file_name": f"{filename}_map.png",
                        "height": tile_size,
                        "width": tile_size,
                        "id": image_id,
                    })

                    sub_vector_xmin = tmp_xmin + x_coord * pixel_size
                    sub_vector_xmax = sub_vector_xmin + tile_size * pixel_size

                    sub_vector_ymax = tmp_ymax - y_coord * pixel_size
                    sub_vector_ymin = sub_vector_ymax - tile_size * pixel_size

                    sub_vector = shapefile.cx[
                        sub_vector_xmin:sub_vector_xmax,
                        sub_vector_ymin:sub_vector_ymax,
                    ]

                    # iterate over the polygons in the goepands vector
                    for _, row in sub_vector.iterrows():
                        polygon = row

                        # get the extens of the current tile
                        x_min_ext = tmp_xmin + x_coord * pixel_size
                        x_max_ext = x_min_ext + tile_size * pixel_size

                        y_min_ext = tmp_ymax - y_coord * pixel_size
                        y_max_ext = y_min_ext - tile_size * pixel_size

                        # get the category for this polygon
                        category = int(polygon["eig_kl_pv"])
                        # get the segmentation for this polygon
                        segmentation_geo = polygon["geometry"].exterior.coords

                        for x, y in segmentation_geo:
                            x = x
                            y = sub_vector_ymax - y

                        full_extent_poly = box(
                            x_min_ext, y_min_ext, x_max_ext, y_max_ext
                        )

                        segmentation_clipped = Polygon(segmentation_geo)\
                            .intersection(full_extent_poly)

                        bounds = segmentation_clipped.bounds

                        if bounds == ():
                            continue

                        bbox_clipped = [
                            bounds[0],
                            bounds[1],
                            bounds[2] - bounds[0],
                            bounds[1] - bounds[3],
                        ]

                        # convert the segmentation to coco format
                        segmentations = []
                        area = 0
                        if segmentation_clipped.geom_type == "Point"\
                                or segmentation_clipped.geom_type == "LineString":
                            continue

                        if segmentation_clipped.geom_type == "MultiPolygon":
                            for segmentation_sub in segmentation_clipped.geoms:
                                segmentation = []
                                for x, y in segmentation_sub.exterior.coords:
                                    x_pix = (x - sub_vector_xmin) / \
                                        pixel_size
                                    y_pix = tile_size - (y - sub_vector_ymin) / \
                                        pixel_size
                                    segmentation += x_pix, y_pix
                                area += segmentation_sub.area / pixel_size ** 2
                                segmentations.append(segmentation)

                        else:
                            segmentation = []
                            for x, y in segmentation_clipped.exterior.coords:
                                x_pix = (x - sub_vector_xmin) / \
                                    pixel_size
                                y_pix = tile_size - (y - sub_vector_ymin) / \
                                    pixel_size
                                segmentation += x_pix, y_pix
                            area = segmentation_clipped.area / pixel_size ** 2
                            segmentations = [segmentation]

                        bbox = [
                            (bbox_clipped[0] - sub_vector_xmin) / pixel_size,
                            tile_size -
                            (bbox_clipped[1] - sub_vector_ymin) / pixel_size,
                            bbox_clipped[2] / pixel_size,
                            bbox_clipped[3] / pixel_size,
                        ]

                        annotation_id = f"{len(coco_json['annotations'])}"
                        coco_json["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category,
                            "segmentation": segmentations,
                            "bbox": bbox,
                            "iscrowd": 0,
                            "area": area,
                        })

                        # keep track of the number of tiles created
                    self.total_tiles += 1

                    # save the coco json file
                    tile_json_path = os.path.join(
                        tile_path,
                        subfolder_fn,
                        f"{map_tile_name}.json",
                    )
                    with open(tile_json_path, "w") as f:
                        json.dump(coco_json, f)

=======
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

>>>>>>> bdb2ba5 (Combine classes into a single roof module)
            print(f"{map_tile_name} tiles created.")
            # when testing, keep temporary files and stop after the first map tile
            if not self._testing:
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path)
            if self._testing:
                break


<<<<<<< HEAD
class DummyExtractor(DataExtractor):
    """ A trusting dummy extractor that simply records what data it finds. """

    def __init__(self, input_path):
        tiles_found = glob.glob(
            os.path.join(input_path, "**", "*_map.png"),
            recursive=True
        )
        self.tile_size = int(os.path.basename(input_path).split("_")[-1])
        self.total_tiles = len(tiles_found)
        self.tile_path = input_path


=======
>>>>>>> bdb2ba5 (Combine classes into a single roof module)
class DataSelector(DataHandler):
    """
    DataSelector handles the selection of data from a folder containing map
    and mask tiles as produced by DataExtractor.
    """

    def __init__(self, extractor: DataExtractor, output_path: str,
                 train_n: int, test_n: int, random_seed=0):
        """Initialize the DataSelector.

        Args:
<<<<<<< HEAD
            extractor (DataExtractor or str): An extractor pointing to the input data.
                Alternatively pass a string pointing to extracted tiles. This will
                use the data pointed to (with a DummyExtractor).
=======
            extractor (DataExtractor): An extractor pointing to the input data.
>>>>>>> bdb2ba5 (Combine classes into a single roof module)
            output_path (str): A path-like in which to store the selected tiles.
                A subdirectory is created within this folder.
            train_n (int): Number of train tiles to select.
            test_n (int): Number of test tiles to select.
            random_seed (int, optional): Seed for shuffling the data. Defaults to 0.

        """
<<<<<<< HEAD

        if type(extractor) is str:
            print("Path was passed so using dummy extractor.")
            self._verify_input_path(extractor)
            self.extractor = DummyExtractor(extractor)
        else:
            self.extractor = extractor
=======
        self.extractor = extractor
>>>>>>> bdb2ba5 (Combine classes into a single roof module)

        self.train_n = train_n
        self.test_n = test_n
        self._verify_request_size()

        self._verify_superdirectory_path(output_path)
        output_subdir = self._subdir_name(
<<<<<<< HEAD
            self.extractor.tile_size,
=======
            extractor.tile_size,
>>>>>>> bdb2ba5 (Combine classes into a single roof module)
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
<<<<<<< HEAD
            output_path=self.output_path,
        )

        self._copy_coco_info(
            file_lists,
=======
>>>>>>> bdb2ba5 (Combine classes into a single roof module)
            input_path=self.extractor.tile_path,
            output_path=self.output_path,
        )

<<<<<<< HEAD
    def _copy_coco_info(self, file_lists, input_path, output_path):

        # get file names into a dict for easier processing
        files = {}
        folders = {}
        folders["train"] = []
        files["train"] = []
        for fn_tuple in file_lists[0]:
            files["train"].append(fn_tuple[0].split("/")[-1])
            folders["train"].append(fn_tuple[0].split("/")[-2])

        folders["test"] = []
        files["test"] = []
        for fn_tuple in file_lists[1]:
            files["test"].append(fn_tuple[0].split("/")[-1])
            folders["test"].append(fn_tuple[0].split("/")[-2])

        for subfolder in ["train", "test"]:

            coco_json = {
                "info": {
                    "description": "",
                    "url": "",
                    "version": "",
                    "year": 2021,
                    "contributor": "",
                    "date_created": "",
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [
                    {
                        "id": 0,
                        "name": "pv1",
                        "supercategory": "roof",
                    },
                    {
                        "id": 1,
                        "name": "pv2",
                        "supercategory": "roof",
                    },
                    {
                        "id": 2,
                        "name": "pv3",
                        "supercategory": "roof",
                    },
                    {
                        "id": 3,
                        "name": "pv4",
                        "supercategory": "roof",
                    },
                ],
            }

            folder_names = []
            for folder_name in folders[subfolder]:
                if folder_name not in folder_names:
                    folder_names.append(folder_name)
            ids = []
            for filename in folder_names:
                all_coco = json.load(
                    open(os.path.join(input_path, filename, filename+".json")))
                for image in all_coco["images"]:
                    if image["file_name"] in files[subfolder]:
                        coco_json["images"].append(image)
                        ids.append(image["id"])
                ids = list(set(ids))
                print(ids)
                for annotation in all_coco["annotations"]:
                    if annotation["image_id"] in ids:
                        annotation["id"] = len(coco_json["annotations"])
                        coco_json["annotations"].append(annotation)
            with open(os.path.join(output_path, subfolder, "coco.json"), "w") as f:
                json.dump(coco_json, f)

=======
>>>>>>> bdb2ba5 (Combine classes into a single roof module)
    def _verify_request_size(self) -> None:
        """Verify that the requested number of tiles can be met.

        Raises:
            InsuffientDataError: If the requested number of tiles cannot be met.
        """
        if self.train_n + self.test_n > self.extractor.total_tiles:
            raise InsuffientDataError(
                f"""Requested {self.train_n} training tiles and {self.test_n} testing tiles,
                but only {self.extractor.total_tiles} tiles available."""
            )

    @classmethod
    def _verify_superdirectory_path(cls, output_path: str) -> None:
        """Verify that the output path exists and is a directory.

        Args:
            output_path (str): A path-like to be checked.
        """
        try:
            cls._verify_output_path(output_path)
        except OutputPathExistsError:
            pass  # ignore that output path exists
        # instead ensure that the output path exists so that we can create subfolders
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        elif not os.path.isdir(output_path):
            raise OutputPathExistsError(
                f"{output_path} exists but is not a directory."
            )

    @staticmethod
    def _subdir_name(tile_size: int, train_n: int, test_n: int, random_seed: int):
        """Create a subdirectory name for the output path."""
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
            random_seed (int): seed for random number generator. Default is 0.

        Returns:
            list: List of lists of tuples pairs of map and mask images
        """
<<<<<<< HEAD
        # get all files in input directory and subdirectories
        files = glob.glob(
            os.path.join(input_path, "**", "*.png"), recursive=True
        )
        files_map = [file for file in files if "map" in file]
        # TODO "mask" is only needed for legacy mode, remove when no longer needed
        files_mask = [file for file in files if "msk" in file or "mask" in file]
=======
        # get all files in input directory
        files = os.listdir(input_path)
        files_map = [file for file in files if "map" in file]
        # TODO "mask" is only needed for legacy mode, remove when no longer needed
        files_mask = [
            file for file in files if "msk" in file or "mask" in file]
>>>>>>> bdb2ba5 (Combine classes into a single roof module)

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
<<<<<<< HEAD
    def _copy_image_files(image_files: list, output_path: str,
                          delete_existing_output_path_no_warning=False):
        """Copy image files from the input path to the output path.

        Args:
            image_files (list): File paths as returned by select_random_map_images
=======
    def _copy_image_files(image_files: list, input_path: str,
                          output_path: str, delete_existing_output_path_no_warning=False):
        """Copy image files from the input path to the output path.

        Args:
            image_files (list): Filenames as returned by select_random_map_images
            input_path (str): original file location
>>>>>>> bdb2ba5 (Combine classes into a single roof module)
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
<<<<<<< HEAD
            raise OutputPathExistsError(
                "At least one of the output directory already exists."
                "\nSet delete_existing=True to remove it."
            )
=======
            raise OutputPathExistsError("At least one of the output directory already exists."
                                        "\nSet delete_existing=True to remove it.")
>>>>>>> bdb2ba5 (Combine classes into a single roof module)

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
<<<<<<< HEAD
                for file_path_in in file_tuple:
                    file_path_out = os.path.join(
                        output_path_subfolder, os.path.basename(file_path_in)
                    )
                    shutil.copy(file_path_in, file_path_out)
=======
                for file_path in file_tuple:
                    full_path_in = os.path.join(input_path, file_path)
                    full_path_out = os.path.join(
                        output_path_subfolder, file_path)
                    shutil.copy(full_path_in, full_path_out)
>>>>>>> bdb2ba5 (Combine classes into a single roof module)
