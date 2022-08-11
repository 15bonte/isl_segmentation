import torch.nn as nn


class TopologyLoss(nn.Module):
    def __init__(self, loss_balance):
        super(TopologyLoss, self).__init__()
        self.loss_balance = loss_balance

    def forward(self, inputs, targets):

        # Topology loss
        topology_input = inputs[:, 0, :, :]
        topology_target = targets[:, 0, :, :]
        topology_loss_function = nn.MSELoss()
        topology_loss = topology_loss_function(topology_input, topology_target)

        # Mask loss
        mask_input = inputs[:, 1, :, :]
        mask_target = targets[:, 1, :, :]
        mask_loss_function = nn.BCEWithLogitsLoss()
        mask_loss = mask_loss_function(mask_input, mask_target)

        return topology_loss + self.loss_balance * mask_loss


class DetailedTopologyLoss:
    def __len__(self):
        return 2

    def __call__(self, inputs, targets):

        # Topology loss
        topology_input = inputs[:, 0, :, :]
        topology_target = targets[:, 0, :, :]
        topology_loss_function = nn.MSELoss()
        topology_loss = topology_loss_function(topology_input, topology_target)

        # Mask loss
        mask_input = inputs[:, 1, :, :]
        mask_target = targets[:, 1, :, :]
        mask_loss_function = nn.BCEWithLogitsLoss()
        mask_loss = mask_loss_function(mask_input, mask_target)

        return [topology_loss, mask_loss]
