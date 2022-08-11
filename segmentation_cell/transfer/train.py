import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch import optim
from segmentation_cell.CellSegmentationParser import CellSegmentationParserTransfer
from segmentation_cell.transfer.model_params import (
    TransferCellSegmentationModelParams,
)
from segmentation_cell.transfer.u_net import TransferNucleusSegmentationUNet
from segmentation_cell.data_loader import SegmentationCellDataLoader
from utils.enum import LossType
from segmentation_cell.CellModelManager import CellModelManager
from utils.TopologyLoss import DetailedTopologyLoss, TopologyLoss


def main(params):
    # Data loading
    loader = SegmentationCellDataLoader(params.data_dir)

    train_dl, val_dl, test_dl, test_add_dl = loader.load_data_set(params)
    if test_dl is None:  # local use
        test_dl = train_dl

    model = TransferNucleusSegmentationUNet(params.model_pretrained_path)
    manager = CellModelManager(model, params, LossType.mAP)

    optimizer = optim.Adam(
        model.parameters(), lr=float(params.learning_rate), betas=(params.beta1, params.beta2),
    )  # define the optimization
    loss_function = TopologyLoss(params.loss_balance)
    detailed_loss_function = DetailedTopologyLoss()
    manager.fit(train_dl, val_dl, optimizer, loss_function, detailed_loss_function)

    manager.predict(test_dl, test_add_dl)


if __name__ == "__main__":
    parser = CellSegmentationParserTransfer()
    args = parser.get_args()

    parameters = TransferCellSegmentationModelParams()
    parameters.update(args)

    main(parameters)
