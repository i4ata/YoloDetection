# https://github.com/pjreddie/darknet/blob/master/cfg/yolov1-tiny.cfg

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=.1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
class FastYOLO1(nn.Module):

    def __init__(self, n_boxes: int = 2, n_classes: int = 20) -> None:
        super().__init__()

        self.n_boxes = n_boxes
        self.n_classes = n_classes

        self.boxes_dims = n_boxes * 5
        self.output_dims = self.boxes_dims + n_classes

        pool = nn.MaxPool2d(2)
        self.net = nn.Sequential(
            ConvBlock(3, 16), pool, 
            ConvBlock(16, 32), pool,
            ConvBlock(32, 64), pool,
            ConvBlock(64, 128), pool,
            ConvBlock(128, 256), pool,
            ConvBlock(256, 512), pool,
            ConvBlock(512, 1024),
            ConvBlock(1024, 256),
            nn.Flatten(),
            nn.Linear(in_features=256 * 7 * 7, out_features= 7 * 7 * self.output_dims)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detections: torch.Tensor = self.net(x)
        detections = detections.view(len(x), 7, 7, self.output_dims)
        detections[..., :self.boxes_dims] = detections[..., :self.boxes_dims].sigmoid() + torch.finfo().eps
        detections[..., self.boxes_dims:] = detections[..., self.boxes_dims:].softmax(-1)
        return detections
