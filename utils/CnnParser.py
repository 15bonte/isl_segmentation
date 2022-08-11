import argparse


class CnnParser:
    """
    CNN parsing class.
    """

    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--data_dir", help="Folder containing .ciz pictures")
        parser.add_argument("--tb_dir", help="Tensorboard folder path")
        parser.add_argument("--model_path", help="Model save path")
        parser.add_argument("--lr", help="Learning rate")
        parser.add_argument("--epochs", help="Number of epochs")
        parser.add_argument("--plot_step", help="Plot every n steps")
        parser.add_argument("--output_dir", help="Folder to save output pictures")
        parser.add_argument(
            "--train_ratio", help="Ratio of input pictures to include in training set"
        )
        parser.add_argument(
            "--validation_ratio", help="Ratio of input pictures to include in validation set"
        )
        parser.add_argument("--test_ratio", help="Ratio of input pictures to include in test set")
        parser.add_argument("--batch_size", help="Batch size")
        parser.add_argument("--dataset_split_folder", help="Dataset split folder")

        self.parser = parser

    def get_args(self):
        return self.parser.parse_args()
