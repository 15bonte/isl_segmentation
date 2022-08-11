from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from bigfish import stack

from utils.enum import NormalizeMethods, ProjectMethods
from utils.preprocessing import move_axis, normalize_array, standardize_array
import copy


class AbstractReader:
    """
    Base class for readers
    """

    def __init__(self, file_path, normalize, mip, dimensions):
        raw_image = stack.read_image(file_path, sanity_check=False)  # Unknown dimensions order

        # For coherence reasons add dimension if image is only 2D
        if len(raw_image.shape) == 2:
            raw_image = np.expand_dims(raw_image, axis=2)

        # For coherence reasons swap axes to have channels at the beginning
        # get argmin of tuple
        channels_axis = np.argmin(raw_image.shape)
        raw_image = np.moveaxis(raw_image, channels_axis, 0)

        # For future operations, avoid uint32 type
        if raw_image.dtype == np.uint32:
            raw_image = raw_image.astype("uint16")

        self.raw_image = raw_image

        self.preprocessing_done = False
        self.processed_image = None

        self.normalize = normalize
        self.mip = mip
        self.dimensions = copy.copy(dimensions)  # avoid consequences of modification in class

        self.file_path = file_path

    def normalize_image(self, image):
        if self.normalize == NormalizeMethods.none:
            return image
        if self.normalize == NormalizeMethods.ZeroAndOne:
            return normalize_array(image)
        if self.normalize == NormalizeMethods.Standardize:
            return standardize_array(image)
        raise ValueError("Unknown normalization method")

    def project_image(self, image):
        if self.mip == ProjectMethods.none:
            return image
        if self.mip == ProjectMethods.Maximum:
            image = stack.maximum_projection(image)
            image = np.expand_dims(image, axis=0)  # DHW, D=1
            return image
        if self.mip == ProjectMethods.Mean:
            image = stack.mean_projection(image)
            image = np.expand_dims(image, axis=0)  # DHW, D=1
            return image
        if self.mip == ProjectMethods.Focus:
            image = stack.focus_projection(image, proportion=0.2)
            image = np.expand_dims(image, axis=0)  # DHW, D=1
            return image
        raise ValueError("Unknown projection method")

    def preprocess_image(self):
        processed_image = np.copy(self.raw_image)
        # Project only if asked
        processed_image = self.project_image(processed_image)
        # Resize image
        resize_dimensions = self.dimensions.get_updated_dimensions(processed_image.shape)
        if resize_dimensions != processed_image.shape:
            processed_image = stack.resize_image(
                processed_image, resize_dimensions, method="bilinear"
            )
        # Normalize
        processed_image = self.normalize_image(processed_image)
        # Swap axes to match HWD format
        if len(processed_image.shape) > 2:
            processed_image = move_axis(processed_image, 0, 2)  # HWD
        # Force float64 type
        if processed_image.dtype == np.uint16:
            processed_image = processed_image * 1.0
        # Store final result
        self.processed_image = np.expand_dims(processed_image, axis=0)  # CHWD (gray-scale)
        self.preprocessing_done = True

    def get_channel(self):
        if not self.preprocessing_done:
            self.preprocess_image()
        return self.processed_image

    @abstractmethod
    def display_info(self):
        pass

    def save(self, destination_path):
        for i in range(self.raw_image.shape[0]):
            plt.imsave(
                destination_path.replace(".", f"_z{i}."), self.raw_image[i, :, :], cmap="gray"
            )

    def display_max_intensity_projection(self):
        plt.figure()
        image = self.get_channel().astype(np.uint16).squeeze()

        plt.title("MIP")
        plt.imshow(image, cmap="gray")

        plt.show()
