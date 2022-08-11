from utils.TransferParser import TransferParser


class NucleusSegmentationParser(TransferParser):
    """
    Nucleus segmentation parsing class.
    """

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            "--slope_factor", help="Slope factor to be used in custom sigmoid function."
        )
