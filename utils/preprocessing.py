import numpy as np
from bigfish import stack


def move_axis(array, original_position, final_position):
    position = original_position
    direction = np.sign(final_position - original_position)
    while position != final_position:
        array = np.swapaxes(array, position, position + direction)
        position = position + direction
    return array


def standardize_array(input_array):
    """
    Expect a 3D array DHW or a 2D array HW
    """

    # 2D
    if len(input_array.shape) == 2:
        return stack.compute_image_standardization(input_array)

    # 3D
    depth = input_array.shape[0]
    output_array = []
    for d in range(depth):
        original_image = input_array[d, :, :]
        normalized_image = stack.compute_image_standardization(original_image)
        output_array.append(normalized_image)
    return np.asarray(output_array)


def normalize_array(input_array):
    """
    Normalize array between 0 and 1
    """
    return (input_array - input_array.min()) / (input_array.max() - input_array.min())
