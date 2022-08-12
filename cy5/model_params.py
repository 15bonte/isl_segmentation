from utils.ModelParams import DataSplit, ModelParams
from utils.dimensions import Dimensions


class Cy5ModelParams(ModelParams):
    """
    Cy5 model params.
    """

    def __init__(self):
        super().__init__("cy5")

        self.dimensions = Dimensions(height=None, width=None)
        self.input_dimensions = Dimensions(height=512, width=512)

        self.split = DataSplit(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1)

        self.num_epochs = 1000
        self.batch_size = 10
        self.learning_rate = 0.1

        self.data_dir = ""

        self.model_load_path = ""
