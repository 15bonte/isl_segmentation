import torch.nn as nn
import torch

from dapi.u_net import DapiUNet


class CustomSigmoid(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return 1 / (1 + torch.exp(-1 * self.alpha * (x - 0.5)))


class TransferCellSegmentationUNet(nn.Module):
    def __init__(self, pretrained_model_path, alpha):
        super().__init__()

        self.u_net = DapiUNet
        self.u_net.load_state_dict(torch.load(pretrained_model_path))
        self.custom_sigmoid = CustomSigmoid(alpha)

    def forward(self, x):
        output = self.u_net(x)
        output = output / 65535.0  # careful, depends on the pretrained model output
        output_test = self.custom_sigmoid(output)
        return output_test
