import torch
from torch.utils.data import Dataset
import numpy as np


class DataSetContainer:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def add_element(self, x_src, y_src):
        self.inputs.append(x_src)
        self.outputs.append(y_src)


class TorchDataset(Dataset):
    """
    Data set to be used for torch models.
    """

    def __init__(self, container):
        super().__init__()

        self.inputs = np.asarray(container.inputs)  # BCHW
        self.outputs = np.asarray(container.outputs)  # BCHW

    # shape of inputs in the dataset
    def __len__(self):
        return self.inputs.shape[0]

    # shape of inputs in the dataset
    def len_x(self):
        return self.inputs.shape

    # shape of outputs in the dataset
    def len_y(self):
        return self.outputs.shape

    # get a row at an index
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.inputs[idx]),
            torch.from_numpy(self.outputs[idx]),
        )

    def reorder(self, mode):
        if mode == "BHWC":
            try:
                self.inputs = self.inputs.transpose((0, 2, 3, 1))
            except:
                pass
            try:
                self.outputs = self.outputs.transpose((0, 2, 3, 1))
            except:
                pass
        else:
            raise ValueError(f"Unknown mode {mode}")
