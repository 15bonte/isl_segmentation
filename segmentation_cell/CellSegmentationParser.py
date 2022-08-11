from utils.CnnParser import CnnParser
from utils.TransferParser import TransferParser


class CellSegmentationParser(CnnParser):
    """
    Cell segmentation parsing class.
    """

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            "--loss_balance", help="Loss balance to be used in topology loss function."
        )


class CellSegmentationParserTransfer(CellSegmentationParser, TransferParser):
    """
    Cell segmentation parsing class.
    """
