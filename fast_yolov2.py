import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
from typing import Literal

from dataset import YOLODataset
from yolo import FastYOLO

def get_anchors(k: int = 5) -> torch.Tensor:
    dataset = YOLODataset()
    all_boxes = torch.cat([dataset[i][1][:, 2:] for i in tqdm(range(len(dataset)))])
    kmeans = KMeans(n_clusters=k, random_state=123, n_init='auto')
    kmeans.fit(all_boxes)
    centers = torch.from_numpy(kmeans.cluster_centers_).float()
    torch.save(centers, 'anchors.pt')
    return centers

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

class FastYOLO2(FastYOLO):
    def __init__(self, n_boxes: int = 2, n_classes: int = 20) -> None:
        super().__init__()
        self.image_size = 416
        self.n_boxes = n_boxes
        self.n_classes = n_classes

        self.anchors = torch.load('anchors.pt') if os.path.exists('anchors.pt') else get_anchors()

        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.net = nn.Sequential(
            ConvBlock(3, 16), pool,
            ConvBlock(16, 32), pool,
            ConvBlock(32, 64), pool,
            ConvBlock(64, 128), pool,
            ConvBlock(128, 256), pool,
            ConvBlock(256, 512), nn.MaxPool2d(kernel_size=2, stride=1),
            ConvBlock(512, 1024),
            ConvBlock(1024, 512),
            nn.Conv2d(512, (n_classes + 5) * len(self.anchors), 1, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detections: torch.Tensor = self.net(x)
        return detections
    
    def _step(self, batch, stage) -> torch.Tensor:
        pass

    def _transform_from_yolo(self, detections) -> torch.Tensor:
        pass

if __name__ == '__main__':
    yolo = FastYOLO2()
    im = torch.rand(2, 3, 416, 416)
    out = yolo(im)
    print(out.shape)
    print(sum(p.numel() for p in yolo.parameters()))