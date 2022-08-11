import os
from dapi.u_net import DapiUNet
from utils.enum import LossType

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from utils.CnnModelManager import CnnModelManager
from torch import optim
from torch import nn
from dapi.data_loader import DapiDataLoader
from dapi.model_params import DapiModelParams
from utils.CnnParser import CnnParser


def main(params):
    # Data loading
    loader = DapiDataLoader(params.data_dir)
    train_dl, val_dl, test_dl, _ = loader.load_data_set(params)
    if test_dl is None:  # local use
        test_dl = train_dl

    # Load pretrained model
    model = DapiUNet
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

    parameters = DapiModelParams()
    parameters.update(args)

    main(parameters)
