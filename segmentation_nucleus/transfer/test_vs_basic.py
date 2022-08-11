import os
from dapi.model_params import DapiModelParams
from dapi.u_net import DapiUNet
from utils.display_tools import display_accuracy

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from utils.CnnModelManager import CnnModelManager
import torch
from segmentation_nucleus.basic.model_params import BasicSegmentationModelParams
from segmentation_nucleus.basic.u_net import BasicSegmentationUNet
from segmentation_nucleus.NucleusSegmentationParser import NucleusSegmentationParser
from segmentation_nucleus.transfer.model_params import TransferSegmentationModelParams
from segmentation_nucleus.transfer.u_net import TransferCellSegmentationUNet
from segmentation_nucleus.data_loader import NucleusSegmentationDataLoader
from utils.enum import LossType


def main(params_transfer, params_basic, params_dapi):
    # Data loading
    loader = NucleusSegmentationDataLoader(params_transfer.data_dir)

    _, _, test_dl, test_add_dl = loader.load_data_set(params_transfer)

    # Transfer model definition
    # Load pretrained transfer model
    model_transfer = TransferCellSegmentationUNet(
        params_transfer.model_pretrained_path, params_transfer.slope_factor
    )
    model_transfer.load_state_dict(torch.load(params_transfer.model_load_path))

    # Predict
    manager_transfer = CnnModelManager(model_transfer, params_transfer, LossType.IoU)
    manager_transfer.predict(test_dl, binary=True)

    # Basic model definition
    # Load pretrained transfer model
    model_basic = BasicSegmentationUNet()
    model_basic.load_state_dict(torch.load(params_basic.model_load_path))

    # Predict
    manager_basic = CnnModelManager(model_basic, params_basic, LossType.IoU)
    manager_basic.predict(test_dl, binary=True)

    # Compare accuracies
    display_accuracy(
        [manager_basic.losses, manager_transfer.losses],
        params_transfer.output_dir,
        LossType.IoU,
        ["Untrained model", "Pretrained model"],
    )

    # Pretask DAPI model definition
    # Load pretrained transfer model
    model_dapi = DapiUNet
    model_dapi.load_state_dict(torch.load(params_dapi.model_load_path))

    # Predict
    manager_dapi = CnnModelManager(model_dapi, params_dapi, LossType.IoU)
    manager_dapi.predict(
        test_dl, test_add_dl=test_add_dl
    )  # ignore precision as ground truth is for another task


if __name__ == "__main__":
    parser = NucleusSegmentationParser()
    args = parser.get_args()

    parameters_transfer = TransferSegmentationModelParams()
    parameters_transfer.update(args)

    parameters_basic = BasicSegmentationModelParams()
    parameters_basic.update(args)

    parameters_dapi = DapiModelParams()
    parameters_dapi.update(args)

    main(parameters_transfer, parameters_basic, parameters_dapi)
