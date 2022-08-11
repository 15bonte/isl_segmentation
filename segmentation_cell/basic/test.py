import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from segmentation_cell.CellModelManager import CellModelManager
import torch
from segmentation_cell.CellSegmentationParser import CellSegmentationParser
from segmentation_cell.basic.model_params import BasicCellSegmentationModelParams
from segmentation_cell.basic.u_net import BasicSegmentationUNet
from segmentation_cell.data_loader import SegmentationCellDataLoader
from utils.enum import LossType


def main(params):
    # Data loading
    loader = SegmentationCellDataLoader(params.data_dir)

    _, _, test_dl, test_add_dl = loader.load_data_set(params)

    # Model definition
    # Load pretrained model
    model = BasicSegmentationUNet()
    model.load_state_dict(torch.load(params.model_load_path))

    manager = CellModelManager(model, params, LossType.mAP)

    manager.predict(test_dl, test_add_dl)


if __name__ == "__main__":
    parser = CellSegmentationParser()
    args = parser.get_args()

    parameters = BasicCellSegmentationModelParams()
    parameters.update(args)

    main(parameters)
