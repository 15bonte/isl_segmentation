import numpy as np
from skimage.segmentation import watershed
from utils.tools import get_local_maxima, simplify_mask


def create_instance_map(mask, topology):
    local_maxima, _ = get_local_maxima(topology)

    # Apply watershed algorithm to recover cell instance
    instance_map = watershed(-topology, local_maxima, mask=mask * 65535)
    return instance_map


def create_instance_map_batch(batch_mask, batch_topology):
    batch_size = batch_mask.shape[0]
    instance_batch = []
    for idx in range(batch_size):
        # Sub select entries to instance map function
        mask = batch_mask[idx, :, :]
        topology = batch_topology[idx, :, :]
        # Instance map creation
        instance = create_instance_map(mask, topology)
        instance = simplify_mask(instance)
        instance_batch.append(instance)
    return np.array(instance_batch)
