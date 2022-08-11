import numpy as np
import random
from skimage.morphology import extrema, reconstruction
from scipy import ndimage


BUFFER_RANDOM = None
BUFFER_RANGE = 0


def random_sample(range_sample, nb_sample):
    global BUFFER_RANGE
    global BUFFER_RANDOM

    if BUFFER_RANDOM is None or nb_sample != len(BUFFER_RANDOM) or range_sample != BUFFER_RANGE:
        BUFFER_RANGE = range_sample
        BUFFER_RANDOM = random.sample(range_sample, nb_sample)
    return BUFFER_RANDOM


def simplify_mask(mask):
    """
    Takes as input instance segmentation image and change unique labels values,
    to range from 0 to 255
    """
    unique_values = np.unique(mask)

    values_dict = {}
    nb_unique = len(unique_values)
    for i, value in enumerate(unique_values):
        new_value = int((i + 1) * 255 / nb_unique)
        if new_value == 0:
            raise ValueError(
                "Too many unique values! A new value is set to 0, which is background."
            )
        values_dict[value] = new_value

    for (i, j), value in np.ndenumerate(mask):
        if value == 0:
            continue
        mask[i, j] = values_dict[value]

    return mask


def compute_i_o_u(ground_truth, prediction):
    flat_gt = ground_truth.flat
    prediction_gt = prediction.flat
    if max(flat_gt) < 0.5 and max(prediction_gt) < 0.5:
        return 1
    # intersection is equivalent to True Positive count
    # union is the mutually inclusive area of all labels & predictions
    intersection = np.multiply(flat_gt, prediction_gt).sum()
    total = np.add(flat_gt, prediction_gt).sum()
    union = total - intersection
    i_o_u = (intersection + 1) / (union + 1)  # 1 for smoothing
    return i_o_u


def get_local_maxima(topology, r_threshold=10, h_threshold=20, footprint=np.ones((3, 3))):
    # h_threshold: Minimum path length to be consider as maxima
    # (cf h_maxima doc)

    float_topology = np.float32(topology)

    seed_topology = float_topology - r_threshold
    reconstructed_topology = reconstruction(seed_topology, float_topology)

    # Local maxima (should be cell centers)
    local_maxima = extrema.h_maxima(reconstructed_topology, h_threshold)
    # Label each maxima pixels cluster with different color
    local_maxima, _ = ndimage.label(local_maxima, structure=footprint)

    return local_maxima, reconstructed_topology
