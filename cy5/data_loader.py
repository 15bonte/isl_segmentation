from random import randrange
from data_set_management.data_sets import DapiDataSet2
from readers.tiff_reader import TiffReader
from utils.AbstractDataLoader import AbstractDataLoader
import numpy as np
from utils.dimensions import Dimensions
from utils.enum import NormalizeMethods
from utils.preprocessing import move_axis
from bigfish import stack


class Cy5DataLoader(AbstractDataLoader, DapiDataSet2):
    """
    Data loader class for UNet model.
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

        # Create reader for output
        cy5_path = self.get_cell_image_path(filename)
        reader_cy5 = TiffReader(cy5_path, dimensions=dimensions,)

        raw_img_input = reader_DIC.get_channel()[:, :, :, ::2]  # C, H, W, D = 1, H, W, 3
        raw_img_input = raw_img_input.squeeze()  # H, W, D = H, W, 3
        raw_img_input = move_axis(raw_img_input, 2, 0)  # D, H, W ~ C, H, W = 3, H, W

        raw_img_output = reader_cy5.get_channel()  # C, H, W, D = 1, H, W, D
        raw_img_output = raw_img_output.squeeze()  # H, W, D
        raw_img_output = move_axis(raw_img_output, 2, 0)  # D, H, W
        raw_img_output = stack.focus_projection(raw_img_output, proportion=0.2)  # H, W
        raw_img_output = np.expand_dims(raw_img_output, 0)  # C, H, W = 1, H, W

        return raw_img_input, raw_img_output, None

    @staticmethod
    def pass_image(image_x, image_y, image_z):
        # If whole image is black then ignore it
        return np.count_nonzero(image_y) == 0

    @staticmethod
    def generate_random_anchor(input_dim, model_dim):
        margin = input_dim.difference(model_dim)

        # Ignore top zone as light is bad (specific to our Data Set 1)
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
