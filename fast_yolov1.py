# https://github.com/pjreddie/darknet/blob/master/cfg/yolov1-tiny.cfg

import torch
import torch.nn as nn

from lightning import LightningModule

from utils import YOLOv1Loss

from typing import Tuple, List

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
    
class FastYOLO1(LightningModule):
    def __init__(self, n_boxes: int = 2, n_classes: int = 20) -> None:
        super(FastYOLO1, self).__init__()
        
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

        self.loss_fn = YOLOv1Loss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detections: torch.Tensor = self.net(x)
        detections = detections.view(len(x), 7, 7, self.output_dims)
        detections[..., :self.boxes_dims].sigmoid_()
        
        return detections
    
    def training_step(self, batch: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        inputs, boxes, labels = batch
        output = self(inputs)
        loss = self.loss_fn(output, boxes, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        inputs, boxes, labels = batch
        output = self(inputs)
        loss = self.loss_fn(output, boxes, labels)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(params=self.net.parameters(), lr=.001)

if __name__ == '__main__':
    yolo = FastYOLO1()
    im = torch.rand(5,3,448,448)
    out = yolo(im)
    print(sum(p.numel() for p in yolo.parameters()))
    print(out.shape)
    print(out[..., :10])
    # print(yolo)