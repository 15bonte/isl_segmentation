import os

from utils.enum import LossType

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from utils.CnnModelManager import CnnModelManager
import torch
from cy5.data_loader import Cy5DataLoader
from cy5.model_params import Cy5ModelParams
from utils.CnnParser import CnnParser
from cy5.u_net import Cy5UNet


def main(params):
    # Data loading
    loader = Cy5DataLoader(params.data_dir)

    _, _, test_dl, _ = loader.load_data_set(params)

    # Model definition
    # Load pretrained model
    model = Cy5UNet
    model.load_state_dict(torch.load(params.model_load_path))

    manager = CnnModelManager(model, params, LossType.PCC)

    manager.predict(test_dl)


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.get_args()

    parameters = Cy5ModelParams()
    parameters.update(args)

    main(parameters)
