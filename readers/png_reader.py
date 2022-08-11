from matplotlib import pyplot as plt
from readers.AbstractReader import AbstractReader
from utils.dimensions import Dimensions
from utils.enum import NormalizeMethods, ProjectMethods


class PngReader(AbstractReader):
    """
    Class to read PNG file.
    """

    def __init__(self, file_path, normalize=NormalizeMethods.none, dimensions=Dimensions()):
        super().__init__(file_path, normalize, ProjectMethods.none, dimensions)

    def display_info(self):
        z_slice_to_plot = self.raw_image.shape[0]
        rows, columns = 1, z_slice_to_plot
        fig = plt.figure()

        for i in range(z_slice_to_plot):
            fig.add_subplot(rows, columns, i + 1)
            plt.title(f"Channel {i}")

            plt.imshow(self.raw_image[i, :, :], cmap="gray")

        plt.show()
