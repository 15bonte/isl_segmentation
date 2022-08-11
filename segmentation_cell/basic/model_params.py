from utils.ModelParams import DataSplit, ModelParams
from utils.dimensions import Dimensions


class BasicCellSegmentationModelParams(ModelParams):
    """
    Basic cell segmentation model params.
    """

    def __init__(self):
        super().__init__("basic_cell_segmentation")

        self.dimensions = Dimensions(height=None, width=None)
        self.input_dimensions = Dimensions(height=512, width=512)

        self.split = DataSplit(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1,)

        self.beta1 = 0.9
        self.beta2 = 0.999

        self.num_epochs = 1000
        self.batch_size = 2
        self.learning_rate = 0.01

        self.loss_balance = 1

        self.data_dir = r"C:\Users\thoma\data\Data Maxence\Data set 2\tiff"

        self.model_load_path = (
            r"C:\Users\thoma\models\20220705-193343\basic_topology_cell_segmentation.pt"
        )

    def update(self, args):
        super().update(args)

        if args.loss_balance:
            self.loss_balance = int(args.loss_balance)
