import matplotlib.pyplot as plt
from readers.AbstractReader import AbstractReader
from utils.dimensions import Dimensions
from PIL import Image
from PIL.TiffTags import TAGS

from utils.enum import NormalizeMethods, ProjectMethods


class TiffReader(AbstractReader):
    """
    Class to read Tiff file.
    Handles 2D or 3D images, coded on 8 or 16bit.
    """

    def __init__(
        self,
        file_path,
        normalize=NormalizeMethods.none,
        mip=ProjectMethods.none,
        dimensions=Dimensions(),
    ):
        super().__init__(file_path, normalize, mip, dimensions)

    def display_info(self):

        with Image.open(self.file_path) as img:
            meta_dict = {TAGS[key]: img.tag[key] for key in img.tag_v2}
            print(meta_dict)

        z_slice_to_plot = self.raw_image.shape[0]
        rows, columns = 1, z_slice_to_plot
        fig = plt.figure()

        for i in range(z_slice_to_plot):
            fig.add_subplot(rows, columns, i + 1)
            plt.title(f"Z{i}")

            plt.imshow(self.raw_image[i, :, :], cmap="gray")

        plt.show()
