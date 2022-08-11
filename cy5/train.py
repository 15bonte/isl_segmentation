import os

from utils.enum import LossType

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from utils.CnnModelManager import CnnModelManager
from torch import optim
from torch import nn
from cy5.data_loader import Cy5DataLoader
from cy5.model_params import Cy5ModelParams
from cy5.u_net import Cy5UNet
from utils.CnnParser import CnnParser
import sys


def main(params):
    # Data loading
    loader = Cy5DataLoader(params.data_dir)
    train_dl, val_dl, test_dl, _ = loader.load_data_set(params)
    if test_dl is None:  # local use
        test_dl = train_dl

    # Load pretrained model
    # Here, 3 channels as inputs: 2 bright-field z-stacks and 1 phase contrast MIP
    model = Cy5UNet

    manager = CnnModelManager(model, params, LossType.PCC)

    optimizer = optim.Adam(
        model.parameters(), lr=float(params.learning_rate), betas=(params.beta1, params.beta2),
    )  # define the optimization
    loss_function = nn.L1Loss()
    manager.fit(train_dl, val_dl, optimizer, loss_function)

    manager.predict(test_dl)


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.get_args()

    parameters = Cy5ModelParams()
    parameters.update(args)

    main(parameters)
