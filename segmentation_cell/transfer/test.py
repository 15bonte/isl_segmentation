import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
from segmentation_cell.CellSegmentationParser import CellSegmentationParserTransfer
from segmentation_cell.transfer.model_params import (
    TransferCellSegmentationModelParams,
)
from segmentation_cell.transfer.u_net import TransferNucleusSegmentationUNet
from segmentation_cell.data_loader import SegmentationCellDataLoader
from utils.enum import LossType
from segmentation_cell.CellModelManager import CellModelManager


def main(params):
    # Data loading
    loader = SegmentationCellDataLoader(params.data_dir)

    _, _, test_dl, test_add_dl = loader.load_data_set(params)

    # Model definition
    # Load pretrained model
    model = TransferNucleusSegmentationUNet(params.model_pretrained_path)
    model.load_state_dict(torch.load(params.model_load_path))

    manager = CellModelManager(model, params, LossType.mAP)

    manager.predict(test_dl, test_add_dl)


if __name__ == "__main__":
    parser = CellSegmentationParserTransfer()
    args = parser.get_args()

    parameters = TransferCellSegmentationModelParams()
    parameters.update(args)

    main(parameters)
