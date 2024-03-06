import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from typing import List, Tuple

def predict_transform(prediction: torch.Tensor, input_dim: int, anchors: List[Tuple[int, int]], num_classes: int) -> torch.Tensor:
    batch_size = len(prediction)
    stride = input_dim // prediction.shape[2] # The factor by which the input image is reduced
    grid_size = input_dim // stride # Size of the grid represented by a cell in the output
    bbox_attributes = 5 + num_classes # x, y, w, h, confidence for an object, confidence for each class 
    num_anchors = len(anchors)

    print(prediction.shape[1:])
    print(bbox_attributes * num_anchors, grid_size ** 2)
    prediction = prediction.view(batch_size, bbox_attributes * num_anchors, grid_size ** 2)
    prediction = prediction.mT.contiguous()
    prediction = prediction.view(batch_size, num_anchors * grid_size ** 2, bbox_attributes)

    anchors = [(a1 / stride, a2 / stride) for a1, a2 in anchors]

    # Normalize x, y, objectness score
    prediction[:, :, [0,1,4]] = torch.sigmoid(prediction[:,:,[0,1,4]])

    # Add the offsets to x and y
    grid = torch.arange(grid_size, device=prediction.device)
    a, b = torch.meshgrid(grid, grid)
    x_offset, y_offset = a.reshape(-1, 1), b.reshape(-1, 1)
    x_y_offset = torch.cat((x_offset, y_offset), dim=1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset

    # Transform w and h
    anchors: torch.Tensor = torch.tensor(anchors, device=prediction.device)
    anchors = anchors.repeat(grid_size ** 2, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Normalize class scores
    prediction[:, :, 5:5+num_classes] = torch.sigmoid(prediction[:, :, 5:5+num_classes])

    # Resize bounding boxes to the size of the image
    prediction[:, :, :4] *= stride

    return prediction
