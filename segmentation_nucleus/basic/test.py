import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from utils.CnnModelManager import CnnModelManager
import torch
from utils.CnnParser import CnnParser
from segmentation_nucleus.basic.model_params import BasicSegmentationModelParams
from segmentation_nucleus.basic.u_net import BasicSegmentationUNet
from segmentation_nucleus.data_loader import NucleusSegmentationDataLoader
from utils.enum import LossType


def main(params):
    # Data loading
    loader = NucleusSegmentationDataLoader(params.data_dir)

    _, _, test_dl, _ = loader.load_data_set(params)

    # Model definition
    # Load pretrained model
    model = BasicSegmentationUNet()
    model.load_state_dict(torch.load(params.model_load_path))

    manager = CnnModelManager(model, params, LossType.IoU)

    manager.predict(test_dl, binary=True)


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.get_args()

    parameters = BasicSegmentationModelParams()
    parameters.update(args)

    main(parameters)
