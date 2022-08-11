import os
import numpy as np
from torch.utils.data import DataLoader
from utils.dimensions import Dimensions
from utils.display_tools import display_progress
from utils.torch_dataset import DataSetContainer, TorchDataset
from abc import abstractmethod
import random as rd

rd.seed(10)


class AbstractDataLoader:
    """
    Data loader abstract class.
    """

    def __init__(self, dir_src, **kwargs):
        super().__init__(**kwargs)  # forwards all unused arguments
        self.dir_src = dir_src

    @abstractmethod
    def generate_raw_images(self, filename, dimensions):
        pass

    @staticmethod
    def pass_image(image_x, image_y, image_z):
        return False

    @staticmethod
    def verify_image_big_enough(image, model_dimension):
        """
        Check if image is not too small
        """
        image_shape = image.shape
        is_3d = len(image_shape) == 4

        if is_3d:
            input_dim = Dimensions(image_shape[1], image_shape[2], image_shape[3])
        else:
            input_dim = Dimensions(image_shape[1], image_shape[2])

        # Protection against wrong dimensions choice
        if model_dimension.is_strict_bigger(input_dim):
            raise ValueError(
                f"Cannot crop {model_dimension.to_tuple()} in smaller {input_dim.to_tuple()}."
            )

        return input_dim

    @staticmethod
    def verify_input_output_equal_sizes(img_x, img_y):
        """
        Verify that input and output image have same size
        """
        # Input and output image must have same dimension to crop inside
        # Ignore first dimension, C, which might be different
        x_shape = img_x.shape[1:]
        y_shape = img_y.shape[1:]
        if x_shape != y_shape:
            raise ValueError(
                f"Input and output images do not have same size: {img_x.shape} vs {img_y.shape}"
            )

    @staticmethod
    def generate_random_anchor(input_dim, model_dim):
        margin = input_dim.difference(model_dim)

        h_start = rd.randrange(margin.height + 1)
        w_start = rd.randrange(margin.width + 1)
        if margin.depth is not None:
            d_start = rd.randrange(margin.depth + 1)
        else:
            d_start = None
        return Dimensions(h_start, w_start, d_start)

    @staticmethod
    def get_crop(img, anchor, output_dim):
        if output_dim.depth is not None:
            return np.copy(
                img[
                    :,
                    anchor.height : anchor.height + output_dim.height,
                    anchor.width : anchor.width + output_dim.width,
                    anchor.depth : anchor.depth + output_dim.depth,
                ]
            )
        return np.copy(
            img[
                :,
                anchor.height : anchor.height + output_dim.height,
                anchor.width : anchor.width + output_dim.width,
            ]
        )

    def generate_sub_images(self, img_x, img_y, img_z, model_dim, n_sub_images):
        """
        Randomly select nb_images sub images of given dimensions in input images.
        Input images should have 4 dimensions: CHWD or 3 dimensions: CHW
        """

        self.verify_input_output_equal_sizes(img_x, img_y)
        input_dim = self.verify_image_big_enough(img_x, model_dim)

        # If model dimension has one field to None then no crop is applied
        if model_dim.has_none(len(img_x.shape) == 4):
            return [img_x], [img_y]

        # z is for additional data
        imgs_x, imgs_y, imgs_z = [], [], []

        # Iteration landmarks
        max_tries = 1000
        current_try = 0
        while len(imgs_y) < n_sub_images and current_try < max_tries:
            current_try += 1
            random_anchor = self.generate_random_anchor(input_dim, model_dim)
            # Crop
            crop_x = self.get_crop(img_x, random_anchor, model_dim)
            crop_y = self.get_crop(img_y, random_anchor, model_dim)
            if img_z is not None:
                crop_z = self.get_crop(img_z, random_anchor, model_dim)
            else:
                crop_z = []
            # Ignore if output image does not respect condition
            if self.pass_image(crop_x, crop_y, crop_z):
                continue
            # Else, add to containers
            imgs_x.append(crop_x)
            imgs_y.append(crop_y)
            imgs_z.append(crop_z)

        if len(imgs_y) < n_sub_images:
            # If not enough acceptable images have been found then ignore whole image,
            # to ensure coherent number of images in train or test data set
            # (e.g. self.n_sub_images * data set length)
            return [], [], []

        return imgs_x, imgs_y, imgs_z

    @staticmethod
    def generate_train_val_test_list(params, files):
        has_folder = params.split.dataset_split_folder != ""
        has_ratios = (
            params.split.train_ratio + params.split.validation_ratio + params.split.test_ratio
        ) > 0

        if has_folder and has_ratios:
            raise ValueError("Both folder and ratios are set, choice is ambiguous.")
        if not has_folder and not has_ratios:
            raise ValueError("Neither folder or ratios are set, what to choose.")

        # Case where data set split is already defined
        if has_folder:
            print(f"Data set split from {params.split.dataset_split_folder}.")
            # Check that needed files exist
            directory = params.split.dataset_split_folder
            train_file = os.path.join(directory, "train.txt")
            val_file = os.path.join(directory, "val.txt")
            test_file = os.path.join(directory, "test.txt")
            files = [train_file, val_file, test_file]
            missing_files = []
            names = []

            # Read files
            for file in files:
                if os.path.isfile(file):
                    with open(file) as f:
                        names.append([line.rstrip() for line in f])
                else:
                    missing_files.append(file.split("\\")[-1])
                    names.append([])

            if len(missing_files) > 0:
                print(f"Warning: file(s) {missing_files} missing in data set split folder.")

        # Case where split is created randomly using given ratios
        else:
            train_ratio, validation_ratio, test_ratio = (
                params.split.train_ratio,
                params.split.validation_ratio,
                params.split.test_ratio,
            )
            names = [[], [], []]

            for file in files:
                # Store in train or test data set
                random_float = rd.random()
                # [0, train_percent] -> train
                # [train_percent, train_percent+val_percent] -> val
                # [train_percent+val_percent, train_percent+val_percent+test_percent] -> test
                # [train_percent+val_percent+test_percent, 1] -> ignore
                if random_float > train_ratio + validation_ratio + test_ratio:
                    continue
                if random_float < train_ratio:
                    names[0].append(file)
                elif random_float < train_ratio + validation_ratio:
                    names[1].append(file)
                else:
                    names[2].append(file)

        return names  # [train, val, test]

    def load_data_set_core(self, params):
        container_train = DataSetContainer()
        container_validation = DataSetContainer()
        container_test = DataSetContainer()
        container_test_add = DataSetContainer()

        print("Data is loading from " + self.dir_src)

        files = self.get_distinct_files()
        files_length = len(files)

        names_train, names_validation, names_test = self.generate_train_val_test_list(
            params, files
        )

        for idx, file in enumerate(files):
            # Check if file will be used for train or test
            to_train = None
            if file in names_train:
                to_train = 1
            elif file in names_validation:
                to_train = 0
            elif file in names_test:
                to_train = -1

            # Neither train or test
            if to_train is None:
                display_progress("Loading in progress", idx + 1, files_length, cpu_memory=True)
                continue

            raw_img_input, raw_img_output, raw_img_add = self.generate_raw_images(
                file, params.dimensions
            )

            if raw_img_input is None or raw_img_output is None:
                display_progress("Loading in progress", idx + 1, files_length, cpu_memory=True)
                continue

            imgs_input, imgs_output, imgs_add = self.generate_sub_images(
                raw_img_input,
                raw_img_output,
                raw_img_add,
                params.input_dimensions,
                params.n_sub_images,
            )

            for img_input, img_output, img_add in zip(imgs_input, imgs_output, imgs_add):
                if to_train == 1:
                    container_train.add_element(img_input, img_output)
                elif to_train == 0:
                    container_validation.add_element(img_input, img_output)
                else:
                    container_test.add_element(img_input, img_output)
                    container_test_add.add_element(img_add, img_add)

            # Loading progress
            display_progress("Loading in progress", idx + 1, files_length, cpu_memory=True)

        dataset_train = TorchDataset(container_train)
        dataset_validation = TorchDataset(container_validation)
        dataset_test = TorchDataset(container_test)
        dataset_test_add = TorchDataset(container_test_add)

        print("\nTrain input shape is (B, C, H, W, Z) = " + str(dataset_train.len_x()))
        print("Train output shape is (B, C, H, W, Z) = " + str(dataset_train.len_y()))
        print("Validation input shape is (B, C, H, W, Z) = " + str(dataset_validation.len_x()))
        print("Validation output shape is (B, C, H, W, Z) = " + str(dataset_validation.len_y()))
        print("Test input shape is (B, C, H, W, Z) = " + str(dataset_test.len_x()))
        print("Test output shape is (B, C, H, W, Z) = " + str(dataset_test.len_y()))

        return dataset_train, dataset_validation, dataset_test, dataset_test_add

    def load_data_set(self, params):
        (
            dataset_train,
            dataset_validation,
            dataset_test,
            dataset_test_add,
        ) = self.load_data_set_core(params)

        train_dl = (
            DataLoader(dataset_train, batch_size=params.batch_size, shuffle=True)
            if dataset_train.__len__()
            else None
        )
        validation_dl = (
            DataLoader(dataset_validation, batch_size=params.batch_size, shuffle=False)
            if dataset_validation.__len__()
            else None
        )
        test_dl = (
            DataLoader(dataset_test, batch_size=params.batch_size, shuffle=False)
            if dataset_test.__len__()
            else None
        )
        test_add_dl = (
            DataLoader(dataset_test_add, batch_size=params.batch_size, shuffle=False)
            if dataset_test_add.__len__()
            else None
        )

        return train_dl, validation_dl, test_dl, test_add_dl
