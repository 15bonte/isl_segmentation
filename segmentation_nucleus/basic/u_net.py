import torch.nn as nn
import segmentation_models_pytorch as smp


class BasicSegmentationUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.u_net = smp.Unet(
            encoder_name="densenet121",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset),
            activation="sigmoid",
        )

    def forward(self, x):
        return self.u_net(x)

