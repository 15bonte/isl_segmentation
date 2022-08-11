import os
import time
import numpy as np
from utils.display_tools import display_accuracy, display_progress
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.enum import LossType

from utils.preprocessing import move_axis
from bigfish import stack
import git
import pathlib

from utils.segmentation_metrics import get_instance_metrics
from utils.tools import compute_i_o_u, random_sample


class CnnModelManager:
    """
    Class with all useful functions to train, test, ... a CNN-based model
    """

    def __init__(self, model, params, loss):
        # Device to train model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.float()
        self.model.to(self.device)

        self.params = params

        self.loss = loss

        # Save n predictions when testing
        self.nb_images_to_save = 20

        # Display current git hash to follow up
        current_file_path = pathlib.Path(__file__).parent.resolve()
        repo = git.Repo(current_file_path, search_parent_directories=True)
        sha = repo.head.object.hexsha
        print(f"Current commit hash: {sha}")

        self.image_index = 0  # used in prediction
        self.losses = []  # used in models comparison

    def fit(self, train_dl, val_dl, optimizer, loss_function, detailed_loss_function=None):
        # Create folder to save model
        os.makedirs(self.params.models_folder, exist_ok=True)

        # Create csv file with all parameters
        csv_path = os.path.join(self.params.models_folder, "parameters.csv")
        with open(csv_path, "w") as f:
            for key in self.params.__dict__.keys():
                f.write("%s,%s\n" % (key, self.params.__dict__[key]))
        f.close()

        epochs = int(self.params.num_epochs)
        num_batches_train = len(train_dl)
        total_batches = num_batches_train * epochs

        # Tensorboard writer
        os.makedirs(self.params.tensorboard_folder_path, exist_ok=True)
        writer = SummaryWriter(self.params.tensorboard_folder_path)
        mode_save_path = f"{self.params.models_folder}/{self.params.model_save_name}"

        # Loss initializer
        if detailed_loss_function is None:
            running_loss = [0.0]
        else:
            running_loss = [0.0 for _ in range(len(detailed_loss_function) + 1)]

        # Monitor training time
        start = time.time()
        best_test_loss = np.Infinity
        model_epoch = -1

        for epoch in range(epochs):
            # enumerate mini batches
            self.model.train()  # set model to train mode
            for i, (inputs, targets) in enumerate(train_dl):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                current_batch = epoch * num_batches_train + i

                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                model_outputs = self.model(inputs.float())
                # calculate loss
                loss = loss_function(model_outputs, targets.float())
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()

                running_loss[0] += loss.item()
                if detailed_loss_function is not None:
                    losses = detailed_loss_function(model_outputs, targets.float())
                    running_loss[1:] = [x + y.item() for x, y in zip(running_loss[1:], losses)]

                plot_step = int(self.params.plot_step)
                if current_batch % plot_step == plot_step - 1:  # every plot_step mini-batches...

                    # ...log the running loss
                    for j, local_loss in enumerate(running_loss):
                        index = "" if j == 0 else j
                        writer.add_scalar(
                            f"training loss{index}", local_loss / plot_step, current_batch
                        )
                        running_loss[j] = 0.0

                display_progress(
                    "Training in progress",
                    current_batch + 1,
                    total_batches,
                    additional_message=f"Local step {i} | Epoch {epoch}",
                    cpu_memory=True,
                )

            evaluate = val_dl is not None
            val_loss = 0
            # If val_dl is not empty then perform evaluation and compute loss
            if evaluate:
                self.model.eval()  # set model to evaluation mode
                with torch.no_grad():
                    for i, (inputs, targets) in enumerate(val_dl):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)

                        # compute the model output
                        model_outputs = self.model(inputs.float())
                        # calculate loss
                        loss = loss_function(model_outputs, targets.float())
                        val_loss += loss.item()
                # ...log the running loss
                writer.add_scalar("validation loss", val_loss / len(val_dl), current_batch)

            # Save only if better than current best loss, or if no evaluation is possible
            if (not evaluate) or (val_loss < best_test_loss):
                torch.save(self.model.state_dict(), mode_save_path)
                best_test_loss = val_loss
                model_epoch = epoch

        end = time.time()
        print(f"\nTraining successfully finished in {end - start}s.")
        print(f"Best model saved at epoch {model_epoch}.")

    def compute_loss(self, ground_truth, prediction, add_data=None):

        if self.loss == LossType.PCC:
            return np.corrcoef(ground_truth.flat, prediction.flat)[0, 1]

        if self.loss == LossType.IoU:
            return compute_i_o_u(ground_truth, prediction)

        if self.loss == LossType.mAP:
            if add_data is None:
                raise ValueError("Additional data cannot be None for instance mAP loss.")

            masks_gt = add_data[0, :, :]
            masks_pred = prediction[-1, :, :]

            instance_metrics = get_instance_metrics(masks_gt, masks_pred)
            return instance_metrics["mean_ap"]

        if self.loss == LossType.ap_50:
            if add_data is None:
                raise ValueError("Additional data cannot be None for instance mAP loss.")

            masks_gt = add_data[0, :, :]
            masks_pred = prediction[-1, :, :]

            instance_metrics = get_instance_metrics(masks_gt, masks_pred)
            return instance_metrics["ap_50"]

        if self.loss == LossType.none:
            return 0

        raise ValueError(f"Unknown loss type {self.loss}")

    @staticmethod
    def model_prediction(model, inputs, adds, binary):
        """
        Function to generate outputs from inputs for given model
        """
        raw_predictions = model(inputs.float())
        numpy_predictions = raw_predictions.detach().cpu().numpy()

        if binary is False:
            return numpy_predictions

        threshold = lambda t: 1 if t > 0.5 else 0.0
        threshold_vec = np.vectorize(threshold)
        predictions = threshold_vec(numpy_predictions)
        return predictions

    def save_results(self, index, input_to_save, ground_truth, prediction, add):
        # Save inputs, targets & predictions as tiff images
        sub_input_to_save = input_to_save[0, :, :]  # only first channel, careful might be specific
        stack.save_image(
            sub_input_to_save, f"{self.params.output_dir}/{index}_input.tiff",
        )

        def channels_at_end(array):
            array_shape = array.shape
            return len(array_shape) and array_shape[-1] < array_shape[-2]

        if channels_at_end(ground_truth):
            ground_truth = move_axis(ground_truth, 2, 0)
        stack.save_image(
            ground_truth, f"{self.params.output_dir}/{index}_ground_truth.tiff",
        )

        if channels_at_end(prediction):
            prediction = move_axis(prediction, 2, 0)
        stack.save_image(prediction, f"{self.params.output_dir}/{index}_predicted.tiff")

        if add is not None:
            if channels_at_end(add):
                add = move_axis(add, 2, 0)
            stack.save_image(add, f"{self.params.output_dir}/{index}_additional.tiff")

    def batch_predict(self, inputs, targets, images_to_save, i, num_batches_test, binary, adds):
        inputs = inputs.to(self.device)
        inputs_to_save = inputs.detach().cpu().numpy()

        if adds is not None:  # adds may be None if no additional data is used for this model
            adds = adds.to(self.device)
            adds_to_save = adds.detach().cpu().numpy()

        ground_truths = targets.detach().numpy()

        # Run prediction, returns numpy array
        predictions = self.model_prediction(self.model, inputs, adds, binary)

        for idx in range(ground_truths.shape[0]):
            input_to_save = inputs_to_save[idx, :, :, :].squeeze()
            ground_truth = ground_truths[idx, :, :, :].squeeze()
            prediction = predictions[idx, :, :, :].squeeze()
            try:
                add_to_save = adds_to_save[idx, :, :, :].squeeze()
            except Exception:
                add_to_save = None

            # Pearson correlation coefficient
            local_loss = self.compute_loss(ground_truth, prediction, add_to_save)

            if self.image_index in images_to_save:
                self.save_results(
                    self.image_index, input_to_save, ground_truth, prediction, add_to_save
                )

            self.image_index += 1

        display_progress(
            "Model evaluation in progress",
            i + 1,
            num_batches_test,
            additional_message=f"Batch #{i}",
        )

        return local_loss

    def predict(self, test_dl, test_add_dl=None, binary=False):
        # Create folder to save predictions
        os.makedirs(self.params.output_dir, exist_ok=True)

        self.model.eval()  # Set eval mode for model

        # Create list with images indexes to save predictions, to avoid saving all
        num_batches_test = len(test_dl)
        total_images = num_batches_test * self.params.batch_size
        nb_images_to_save = min(total_images, self.nb_images_to_save)
        images_to_save = random_sample(range(total_images), nb_images_to_save)

        losses = []

        with torch.no_grad():

            # Use trained model to predict on test set
            if test_add_dl is None:
                for i, (inputs, targets) in enumerate(test_dl):
                    local_loss = self.batch_predict(
                        inputs, targets, images_to_save, i, num_batches_test, binary, None
                    )
                    losses.append(local_loss)
            else:
                for i, ((inputs, targets), (adds, _)) in enumerate(zip(test_dl, test_add_dl)):
                    local_loss = self.batch_predict(
                        inputs, targets, images_to_save, i, num_batches_test, binary, adds
                    )
                    losses.append(local_loss)

        self.losses = losses  # to be used in models comparison
        # If we compare two models then force scale to be from 0 to 1, otherwise let it automatic
        display_accuracy([losses], self.params.output_dir, self.loss, [self.params.name])
