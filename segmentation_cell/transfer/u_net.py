import torch.nn as nn
import segmentation_models_pytorch as smp
import torch


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding="same")
        self.batch = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batch(self.conv(x)))


class TransferNucleusSegmentationUNet(nn.Module):
    def __init__(self, pretrained_model_path):
        super().__init__()

        self.u_net = smp.Unet(
            encoder_name="densenet121",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
        self.u_net.load_state_dict(torch.load(pretrained_model_path))
        # Preprocess output before sending it to sigmoid function (already in loss)
        self.block_1 = Block(1, 2)
        self.block_2 = Block(2, 2)

    def forward(self, x):
        output = self.u_net(x)
        output = output / 65535.0  # careful, depends on the pretrained model output
        output = self.block_1(output)
        output = self.block_2(output)
        return output
