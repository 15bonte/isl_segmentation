import os
from utils.enum import LossType


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch import optim
from utils.CnnParser import CnnParser
from utils.CnnModelManager import CnnModelManager
from segmentation_nucleus.basic.model_params import BasicSegmentationModelParams
from segmentation_nucleus.basic.u_net import BasicSegmentationUNet
from segmentation_nucleus.data_loader import NucleusSegmentationDataLoader
from utils.IoULoss import IoULoss


def main(params):
    # Data loading
    loader = NucleusSegmentationDataLoader(params.data_dir)

    train_dl, val_dl, test_dl, _ = loader.load_data_set(params)
    if test_dl is None:  # local use
        test_dl = train_dl

    model = BasicSegmentationUNet()
    manager = CnnModelManager(model, params, LossType.IoU)

    optimizer = optim.Adam(
        model.parameters(), lr=float(params.learning_rate), betas=(params.beta1, params.beta2),
    )  # define the optimization
    loss_function = IoULoss()
    manager.fit(train_dl, val_dl, optimizer, loss_function)

    manager.predict(test_dl, binary=True)


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.get_args()

    parameters = BasicSegmentationModelParams()
    parameters.update(args)

    main(parameters)
