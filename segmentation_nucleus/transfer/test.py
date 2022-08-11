import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from utils.CnnModelManager import CnnModelManager
import torch
from segmentation_nucleus.NucleusSegmentationParser import NucleusSegmentationParser
from segmentation_nucleus.transfer.model_params import TransferSegmentationModelParams
from segmentation_nucleus.transfer.u_net import TransferCellSegmentationUNet
from segmentation_nucleus.data_loader import NucleusSegmentationDataLoader
from utils.enum import LossType


def main(params):
    # Data loading
    loader = NucleusSegmentationDataLoader(params.data_dir)

    _, _, test_dl, _ = loader.load_data_set(
        params=params, train_ratio=0.0, validation_ratio=0.0, test_ratio=1.0
    )

    # Model definition
    # Load pretrained model
    model = TransferCellSegmentationUNet(params.model_pretrained_path, params.slope_factor)
    model.load_state_dict(torch.load(params.model_load_path))

    manager = CnnModelManager(model, params, LossType.IoU)

    manager.predict(test_dl, binary=True)


if __name__ == "__main__":
    parser = NucleusSegmentationParser()
    args = parser.get_args()

    parameters = TransferSegmentationModelParams()
    parameters.update(args)

    main(parameters)
