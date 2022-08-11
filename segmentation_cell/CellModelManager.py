from utils.CnnModelManager import CnnModelManager
from bigfish import stack
import numpy as np
import torch.nn as nn

from utils.Topology2Instance import create_instance_map_batch


class CellModelManager(CnnModelManager):
    @staticmethod
    def model_prediction(model, inputs, adds, binary):
        raw_predictions = model(inputs.float())  # B, C, H, W = B, 2, H, W

        # Topology prediction
        topology_predictions = raw_predictions[:, 0, :, :]  # B, H, W
        np_topology_predictions = topology_predictions.detach().cpu().numpy()

        # Mask predicition
        mask_predictions = raw_predictions[:, 1, :, :]  # B, H, W
        mask_predictions = nn.Sigmoid()(mask_predictions)
        np_mask_predictions = mask_predictions.detach().cpu().numpy()

        threshold = lambda t: 1 if t > 0.5 else 0.0
        threshold_vec = np.vectorize(threshold)
        np_mask_predictions = threshold_vec(np_mask_predictions)

        # Instance prediction
        np_instance_predictions = create_instance_map_batch(
            np_mask_predictions, np_topology_predictions
        )

        final_prediction = np.concatenate(
            (
                np.expand_dims(np_topology_predictions, axis=1),
                np.expand_dims(np_mask_predictions, axis=1),
                np.expand_dims(np_instance_predictions, axis=1),
            ),
            axis=1,
        )  # B, C, H, W = B, 3, H, W

        return final_prediction

    def save_results(self, index, input_to_save, ground_truth, prediction, add):
        # Save inputs, targets & predictions as png images
        sub_input_to_save = input_to_save[0, :, :]  # only first channel, careful might be specific
        stack.save_image(
            sub_input_to_save, f"{self.params.output_dir}/{index}_input.tiff",
        )

        ground_truth_topology = ground_truth[0, :, :].astype(np.uint8)
        stack.save_image(
            ground_truth_topology, f"{self.params.output_dir}/{index}_ground_truth_topology.tiff",
        )
        ground_truth_mask = (65535 * ground_truth[1, :, :]).astype(
            np.uint8
        )  # *65535 because from CellPose
        stack.save_image(
            ground_truth_mask, f"{self.params.output_dir}/{index}_ground_truth_mask.tiff",
        )

        predicted_topology = prediction[0, :, :].astype(np.uint8)
        stack.save_image(
            predicted_topology, f"{self.params.output_dir}/{index}_predicted_topology.tiff",
        )
        predicted_mask = (255 * prediction[1, :, :]).astype(np.uint8)
        stack.save_image(
            predicted_mask, f"{self.params.output_dir}/{index}_predicted_mask.tiff",
        )
        predicted_instance = prediction[2, :, :].astype(np.uint8)
        stack.save_image(
            predicted_instance, f"{self.params.output_dir}/{index}_predicted_instance.tiff",
        )

        ground_truth_instance = add[0, :, :].astype(np.uint8)
        stack.save_image(
            ground_truth_instance, f"{self.params.output_dir}/{index}_ground_truth_instance.tiff",
        )
        ground_truth_nucleus = (add[1, :, :] * 65535).astype(np.uint8)
        stack.save_image(
            ground_truth_nucleus, f"{self.params.output_dir}/{index}_ground_truth_nucleus.tiff",
        )  # *65535 because from CellPose
