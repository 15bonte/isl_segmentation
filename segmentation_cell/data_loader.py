from random import randrange
from data_set_management.data_sets import DapiDataSet2
from readers.tiff_reader import TiffReader
from utils.AbstractDataLoader import AbstractDataLoader
import numpy as np
from utils.dimensions import Dimensions
from utils.enum import NormalizeMethods

from utils.preprocessing import move_axis
from bigfish import stack

from utils.tools import simplify_mask


class SegmentationCellDataLoader(AbstractDataLoader, DapiDataSet2):
    """
    Data loader class for cell segmentation model.
    """

    def __init__(self, dir_src):
        super().__init__(data_set_dir=dir_src, dir_src=dir_src)  # forwards all unused arguments

    def generate_raw_images(self, filename, dimensions):
        # w1Cy3 ; w2GFP ; w3Hoechst ; w4Cy5 ; w5DIC

        # Create reader for input
        dic_path = self.get_dic_image_path(filename)
        reader_DIC = TiffReader(
            dic_path, normalize=NormalizeMethods.Standardize, dimensions=dimensions,
        )

        raw_img_input = reader_DIC.get_channel()[:, :, :, ::2]  # C, H, W, D = 1, H, W, 3
        raw_img_input = raw_img_input.squeeze()  # H, W, D = H, W, 3
        raw_img_input = move_axis(raw_img_input, 2, 0)  # D, H, W ~ C, H, W = 3, H, W

        # Read output
        cell_topology_path = self.get_cell_topology_image_path(filename)
        raw_img_topology = stack.read_image(cell_topology_path)  # H, W
        raw_img_topology = np.expand_dims(raw_img_topology, 0)  # C, H, W = 1, H, W

        cell_mask_path = self.get_cell_semantic_image_path(filename)
        raw_img_mask = stack.read_image(cell_mask_path)  # H, W
        raw_img_mask = np.expand_dims(raw_img_mask, 0)  # C, H, W = 1, H, W
        raw_img_mask = raw_img_mask / 65535.0  # careful here, considering cellpose to be uint16

        raw_img_output = np.concatenate(
            (raw_img_topology, raw_img_mask), axis=0
        )  # C, H, W = 2, H, W

        # Read additional data
        cell_instance_path = self.get_cell_instance_image_path(filename)
        raw_img_instance = stack.read_image(cell_instance_path)  # H, W
        raw_img_instance = simplify_mask(raw_img_instance)
        raw_img_instance = np.expand_dims(raw_img_instance, 0)  # C, H, W = 1, H, W

        nucleus_instance_path = self.get_nucleus_instance_image_path(filename)
        raw_img_nucleus = stack.read_image(nucleus_instance_path)  # H, W
        raw_img_nucleus = (
            raw_img_nucleus / 65535.0
        )  # careful here, considering cellpose to be uint16
        raw_img_nucleus = np.expand_dims(raw_img_nucleus, 0)  # C, H, W = 1, H, W

        cy5_path = self.get_cell_image_path(filename)
        reader_cy5 = TiffReader(cy5_path, dimensions=dimensions,)
        raw_img_cy5 = reader_cy5.get_channel()[:, :, :, -1]  # C, H, W, D = 1, H, W, 1
        raw_img_cy5 = raw_img_cy5.squeeze()  # H, W
        raw_img_cy5 = np.expand_dims(raw_img_cy5, 0)  # C, H, W = 1, H, W

        raw_img_add = np.concatenate(
            (raw_img_instance, raw_img_nucleus, raw_img_cy5), axis=0
        )  # C, H, W = 2, H, W

        return raw_img_input, raw_img_output, raw_img_add

    @staticmethod
    def pass_image(image_x, image_y, image_z):
        # If whole image is black then ignore it
        return np.count_nonzero(image_y) == 0

    @staticmethod
    def generate_random_anchor(input_dim, model_dim):
        margin = input_dim.difference(model_dim)

        # Ignore top zone as light is bad (specific to our Data Set 2)
        minimum_height = int(input_dim.height / 3)
        if minimum_height > margin.height:
            raise ValueError(
                f"Not possible to crop with height {margin.height} if top {minimum_height}px are forbidden."
            )

        h_start = randrange(minimum_height, margin.height)
        w_start = randrange(margin.width)
        if margin.depth is not None:
            d_start = randrange(margin.depth)
        else:
            d_start = None
        return Dimensions(h_start, w_start, d_start)
