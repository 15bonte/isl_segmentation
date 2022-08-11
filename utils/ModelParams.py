from datetime import datetime
import os
from utils.dimensions import Dimensions


class DataSplit:
    """
    Split the dataset into train, validation and test.
    """

    def __init__(self, train_ratio=0, validation_ratio=0, test_ratio=0, dataset_split_folder=""):
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.dataset_split_folder = dataset_split_folder


class ModelParams:
    """
    Model params base class.
    """

    def __init__(self, name):
        self.name = name

        # Dimensions used to resize picture to have same scale over all axes
        self.dimensions = Dimensions(height=None, width=None, depth=None)
        # Input dimensions to the model
        self.input_dimensions = Dimensions(height=None, width=None, depth=None)
        # Each input images should generate n_sub_images images
        self.n_sub_images = 5

        # Ratios to split data set
        self.split = DataSplit(
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
            # dataset_split_folder=(
            #     r"C:\Users\thoma\data\Data Maxence\Data set 2\data set split cell\split_1"
            # ),
        )

        # Adam optimizer
        self.beta1 = 0.5
        self.beta2 = 0.999

        # Training parameters
        self.num_epochs = 50
        self.batch_size = 10
        self.learning_rate = 0.1

        # Input data set
        self.data_dir = ""

        # Tensorboard parameters
        self.tensorboard_folder_path = r"C:\Users\thoma\tensorboard\local"
        self.plot_step = 10

        # Output folders - models & predictions
        self.models_folder = r"C:\Users\thoma\models\local"
        self.model_save_name = f"{self.name}.pt"
        self.output_dir = r"C:\Users\thoma\predictions\local"

        # Path to load model to predict
        self.model_load_path = ""

        # Path to folder containing dataset repartition
        self.dataset_split_folder = ""

    def update(self, args):
        if args.data_dir:
            self.data_dir = args.data_dir
        if args.tb_dir:
            self.tensorboard_folder_path = args.tb_dir
        if args.model_path:
            self.models_folder = args.model_path
        if args.lr:
            self.learning_rate = float(args.lr)
        if args.epochs:
            self.num_epochs = int(args.epochs)
        if args.plot_step:
            self.plot_step = int(args.plot_step)
        if args.train_ratio:
            self.split.train_ratio = float(args.train_ratio)
        if args.validation_ratio:
            self.split.validation_ratio = float(args.validation_ratio)
        if args.test_ratio:
            self.split.test_ratio = float(args.test_ratio)
        if args.output_dir:
            self.output_dir = args.output_dir
        if args.batch_size:
            self.batch_size = int(args.batch_size)
        if args.dataset_split_folder:
            self.split.dataset_split_folder = args.dataset_split_folder

        # Create folders dedicated to current run
        now = datetime.now()
        format_now = now.strftime("%Y%m%d-%H%M%S")

        print(f"Model time id: {format_now}")
        print(
            f"epochs {self.num_epochs} | batch {self.batch_size} | lr {self.learning_rate} | plot step {self.plot_step}"
        )

        self.tensorboard_folder_path = f"{self.tensorboard_folder_path}/{format_now}_{self.name}"

        self.models_folder = f"{self.models_folder}/{format_now}"

        self.output_dir = f"{self.output_dir}/{self.name}/{format_now}"
