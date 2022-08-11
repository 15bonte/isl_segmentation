import os

from utils.display_tools import display_accuracy

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from utils.CnnModelManager import CnnModelManager
import torch
from segmentation_cell.basic.model_params import BasicCellSegmentationModelParams
from segmentation_cell.basic.u_net import BasicSegmentationUNet
from segmentation_cell.transfer.model_params import (
    TransferCellSegmentationModelParams,
)
from segmentation_cell.transfer.u_net import TransferNucleusSegmentationUNet
from segmentation_cell.data_loader import SegmentationCellDataLoader
from utils.enum import LossType
import segmentation_models_pytorch as smp
from cy5.model_params import Cy5ModelParams
from segmentation_cell.CellModelManager import CellModelManager
from segmentation_cell.CellSegmentationParser import CellSegmentationParserTransfer


def main(params_transfer, params_basic, params_cy5):
    # Data loading
    loader = SegmentationCellDataLoader(params_transfer.data_dir)

    _, _, test_dl, test_add_dl = loader.load_data_set(params_transfer)

    # Transfer model definition
    # Load pretrained transfer model
    model_transfer = TransferNucleusSegmentationUNet(params_transfer.model_pretrained_path)
    model_transfer.load_state_dict(torch.load(params_transfer.model_load_path))

    # Predict
    manager_transfer = CellModelManager(model_transfer, params_transfer, LossType.mAP)
    manager_transfer.predict(test_dl, test_add_dl, binary=True)

    # Basic model definition
    # Load pretrained transfer model
    model_basic = BasicSegmentationUNet()
    model_basic.load_state_dict(torch.load(params_basic.model_load_path))

    # Predict
    manager_basic = CellModelManager(model_basic, params_basic, LossType.mAP)
    manager_basic.predict(test_dl, test_add_dl, binary=True)

    # Compare accuracies
    display_accuracy(
        [manager_basic.losses, manager_transfer.losses],
        params_transfer.output_dir,
        LossType.mAP,
        ["Untrained model", "Pretrained model"],
    )

    # Pretask Cy5 model definition
    # Load pretrained transfer model
    model_cy5 = smp.Unet(
        encoder_name="densenet121",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
    model_cy5.load_state_dict(torch.load(params_cy5.model_load_path))

    # Predict
    manager_cy5 = CnnModelManager(model_cy5, params_cy5, loss=LossType.none)
    manager_cy5.predict(
        test_dl, test_add_dl
    )  # ignore precision as ground truth is for another task


if __name__ == "__main__":
    parser = CellSegmentationParserTransfer()
    args = parser.get_args()

    parameters_transfer = TransferCellSegmentationModelParams()
    parameters_transfer.update(args)

    parameters_basic = BasicCellSegmentationModelParams()
    parameters_basic.update(args)

    parameters_cy5 = Cy5ModelParams()
    parameters_cy5.update(args)

    main(parameters_transfer, parameters_basic, parameters_cy5)
