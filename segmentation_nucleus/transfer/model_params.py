from utils.ModelParams import DataSplit, ModelParams
from utils.dimensions import Dimensions


class TransferSegmentationModelParams(ModelParams):
    """
    Transfer learning nucleus segmentation model params.
    """

    def __init__(self):
        super().__init__("transfer_nucleus_segmentation")

        self.dimensions = Dimensions(height=None, width=None)
        self.input_dimensions = Dimensions(height=512, width=512)

        self.split = DataSplit(
            train_ratio=1.0,
            test_ratio=0.0,
            # dataset_split_folder=(
            #     r"C:\Users\thoma\data\Data Maxence\Data set 1\data set split\split_1"
            # ),
        )

        self.beta1 = 0.9
        self.beta2 = 0.999

        self.num_epochs = 100
        self.batch_size = 2
        self.learning_rate = 0.01

        self.slope_factor = 1

        self.data_dir = r"C:\Users\thoma\data\Data Maxence\Data set 1\tiff"

        self.model_load_path = r"C:\Users\thoma\models\20220706-110447\transfer_segmentation.pt"

        # Path to DAPI trained model
        self.model_pretrained_path = r"C:\Users\thoma\models\20220421-140340\maxence.pt"

    def update(self, args):
        super().update(args)

        if args.model_pretrained_path:
            self.model_pretrained_path = args.model_pretrained_path

        if args.slope_factor:
            self.slope_factor = int(args.slope_factor)
