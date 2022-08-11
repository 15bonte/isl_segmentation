from utils.ModelParams import DataSplit, ModelParams
from utils.dimensions import Dimensions


class DapiModelParams(ModelParams):
    """
    DAPI model params.
    """

    def __init__(self):
        super().__init__("dapi")

        self.dimensions = Dimensions(height=None, width=None)
        self.input_dimensions = Dimensions(height=512, width=512)

        self.split = DataSplit(train_ratio=0.01, test_ratio=0.01)

        self.num_epochs = 50
        self.batch_size = 2
        self.learning_rate = 0.1

        self.data_dir = r"C:\Users\thoma\data\Data Maxence\Data set 1\tiff"

        self.model_load_path = r"C:\Users\thoma\top models\20220506-120559-maxence.pt"
