import os
from dapi.u_net import DapiUNet
from utils.enum import LossType

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from utils.CnnModelManager import CnnModelManager
import torch
from dapi.data_loader import DapiDataLoader
from dapi.model_params import DapiModelParams
from utils.CnnParser import CnnParser


def main(params):
    # Data loading
    loader = DapiDataLoader(params.data_dir)

    _, _, test_dl, _ = loader.load_data_set(params)

    # Model definition
    # Load pretrained model
    model = DapiUNet
    model.load_state_dict(torch.load(params.model_load_path))

    manager = CnnModelManager(model, params, LossType.PCC)

    manager.predict(test_dl)


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.get_args()

    parameters = DapiModelParams()
    parameters.update(args)

    main(parameters)
