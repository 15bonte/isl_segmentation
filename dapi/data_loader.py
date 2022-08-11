from data_set_management.data_sets import DapiDataSet1
from readers.tiff_reader import TiffReader
from utils.AbstractDataLoader import AbstractDataLoader
import numpy as np
from utils.enum import NormalizeMethods, ProjectMethods

from utils.preprocessing import move_axis
from bigfish import segmentation


class DapiDataLoader(AbstractDataLoader, DapiDataSet1):
    """
    Data loader class for UNet model.
    """

    def __init__(self, dir_src):
        super().__init__(data_set_dir=dir_src, dir_src=dir_src)  # forwards all unused arguments

    def generate_raw_images(self, filename, dimensions):
        # ch1 = Contrast ; ch2 = Cy3 ; ch3 = GFP ; ch4 = DAPI ; ch5 = BrightField
        # Create reader for input
        bright_field_path = self.get_bright_field_image_path(filename)
        reader_bright_field = TiffReader(
            bright_field_path, normalize=NormalizeMethods.Standardize, dimensions=dimensions,
        )

        phase_contrast_path = self.get_phase_contrast_image_path(filename)
        reader_contrast = TiffReader(
            phase_contrast_path,
            normalize=NormalizeMethods.Standardize,
            mip=ProjectMethods.Maximum,
            dimensions=dimensions,
        )

        # Create reader for output
        dapi_path = self.get_dapi_image_path(filename)
        reader_output = TiffReader(dapi_path, mip=ProjectMethods.Maximum, dimensions=dimensions,)

        raw_img_bright = reader_bright_field.get_channel()[:, :, :, 1:3]  # C, H, W, D = 1, H, W, 2
        raw_img_bright = raw_img_bright.squeeze()  # H, W, D = H, W, 2
        raw_img_bright = move_axis(raw_img_bright, 2, 0)  # D, H, W ~ C, H, W = 2, H, W

        raw_img_contrast = reader_contrast.get_channel()  # C, H, W, D = 1, H, W, 1
        raw_img_contrast = raw_img_contrast.squeeze()  # H, W
        raw_img_contrast = np.expand_dims(raw_img_contrast, 0)  # C, H, W = 1, H, W

        raw_img_input = np.concatenate(
            (raw_img_bright, raw_img_contrast), axis=0
        )  # C, H, W = 3, H, W

        raw_img_output = reader_output.get_channel()  # C, H, W, D = 1, H, W, 1
        raw_img_output = raw_img_output.squeeze()  # H, W
        raw_img_output = np.expand_dims(raw_img_output, 0)  # C, H, W = 1, H, W

        return raw_img_input, raw_img_output, None

    @staticmethod
    def pass_image(image_x, image_y, image_z):
        # Test if crop contains nucleus or not
        segmented_input = segmentation.thresholding(
            image_y.squeeze().astype(np.uint16), threshold=5000
        )
        # If no cell is found then ignore current crop
        return np.all(segmented_input == False)
