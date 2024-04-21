# https://github.com/pjreddie/darknet/blob/master/cfg/yolov1-tiny.cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, box_convert
from typing import Tuple, List, Literal

from yolo import FastYOLO

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
    
class YOLOv1Loss(nn.Module):
    def __init__(self, lambda_coord: float = 5., lambda_noobj: float = .5) -> None:
        super(YOLOv1Loss, self).__init__()
        
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, obj_mask: torch.Tensor) -> torch.Tensor:
        """
        y_pred shape: [batch_size, 7, 7, [cx,cy,w,h,c] + C]
        y_true_shape: [batch_size, 7, 7, [cx,cy,w,h,c,i]], where i is the index of the class
        """

        # Get the noobj mask
        noobj_mask = ~obj_mask

        return (
            # x
            self.lambda_coord * F.mse_loss(input=y_pred[..., 0][obj_mask], target=y_true[..., 0][obj_mask]) + 
            # y
            self.lambda_coord * F.mse_loss(input=y_pred[..., 1][obj_mask], target=y_true[..., 1][obj_mask]) + 
            # w
            self.lambda_coord * F.mse_loss(input=y_pred[..., 2][obj_mask].sqrt(), target=y_true[..., 2][obj_mask].sqrt()) + 
            # h
            self.lambda_coord * F.mse_loss(input=y_pred[..., 3][obj_mask].sqrt(), target=y_true[..., 3][obj_mask].sqrt()) + 
            # c_obj
            F.mse_loss(input=y_pred[..., 4][obj_mask], target=y_true[..., 4][obj_mask]) + 
            # c_noobj
            self.lambda_noobj * F.mse_loss(input=y_pred[..., 4][noobj_mask], target=y_true[..., 4][noobj_mask]) + 
            # C
            F.mse_loss(input=y_pred[..., 5:][obj_mask], target=F.one_hot(y_true[..., 5][obj_mask].long(), num_classes=y_pred.size(-1) - 5).float())
        )

class FastYOLO1(FastYOLO):

    def __init__(self, n_boxes: int = 2, n_classes: int = 20) -> None:
        super().__init__()
        
        self.image_size = 448

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
        self.iou_func = torch.vmap(torch.vmap(torch.vmap(box_iou)))
        self.box_convert_func = torch.vmap(torch.vmap(torch.vmap(box_convert)))

        # This is a [7,7,2] grid, where the [i,j]-th cell is equal to [i,j]
        self.grid = torch.stack(torch.meshgrid(torch.arange(7), torch.arange(7), indexing='ij'), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detections: torch.Tensor = self.net(x)
        detections = detections.view(len(x), 7, 7, self.output_dims)
        detections[..., :self.boxes_dims] = detections[..., :self.boxes_dims].sigmoid()
        detections[..., self.boxes_dims:] = detections[..., self.boxes_dims:].softmax(-1)
        
        return detections
    
    def _process_output(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Extract the bounding boxes [batch_size, S, S, B, 5]
        boxes = y_pred[..., :self.boxes_dims].view(*y_pred.shape[:-1], self.n_boxes, 5)

        # Get the obj_mask
        obj_mask = y_true[..., 4] != 0

        # For each cell, get the ious between the predicted bounding boxes and the true one [batch_size, S, S, 5]
        ious: torch.Tensor = self.iou_func(
            self.box_convert_func(boxes[..., :4], in_fmt='cxcywh', out_fmt='xyxy'),
            self.box_convert_func(y_true[..., :4].unsqueeze(-2), in_fmt='cxcywh', out_fmt='xyxy')
        ).squeeze(-1)

        # For each cell, get the largest iou and the largest index ([batch_size, S, S], [batch_size, S, S])
        max_ious, best_ious = ious.max(dim=-1)

        # Compute the true bounding boxes (the ones with the highest iou) [batch_size, S, S, 5]
        detections = torch.gather(
            input=boxes, 
            dim=3, 
            index=best_ious.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,-1,5)
        ).squeeze(-2)

        # The model predicts the IOU
        y_true[..., 4] *= max_ious

        detections = torch.cat((detections, y_pred[..., self.boxes_dims:]), dim=-1)
        return detections, y_true, obj_mask

    def _step(self, batch:  Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]], stage: Literal['train', 'val']) -> torch.Tensor:
        inputs, target_boxes, target_labels = batch
        y_pred: torch.Tensor = self(inputs)

        # Transform targets to yolo coordinates
        y_true = torch.stack(
            [self._transform_to_yolo(boxes=boxes, labels=labels) for boxes, labels in zip(target_boxes, target_labels)]
        )

        detections, y_true, obj_mask = self._process_output(y_pred=y_pred, y_true=y_true)

        loss = self.loss_fn(detections, y_true, obj_mask)
        self.losses[stage].append(loss)
        
        self._get_map(detections=detections, target_boxes=target_boxes, target_labels=target_labels, stage=stage)
        
        return loss
    
    def _transform_from_yolo(self, detections: torch.Tensor) -> torch.Tensor:
        detections[..., [0,1]] += self.grid.to(detections.device)
        detections[..., [0,1]] *= 64
        detections[..., [2,3]] *= 448
        return detections.flatten(start_dim=1, end_dim=2)

    def _transform_to_yolo(self, boxes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        x, y, w, h = boxes.T
        x, y = x / 64., y / 64.
        w, h = w / 448., h / 448.

        objectness_score = torch.ones(len(boxes), device=boxes.device)

        feature_map = torch.zeros(7, 7, 6, device=boxes.device)
        feature_map[x.long(), y.long()] = torch.stack(
            (x.frac(), y.frac(), w, h, objectness_score, labels), dim=1
        )

        return feature_map

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(params=self.net.parameters(), lr=1e-3, momentum=.9, weight_decay=5e-4)

if __name__ == '__main__':
    yolo = FastYOLO1()
    im = torch.rand(5,3,448,448)
    out = yolo(im)
    print(sum(p.numel() for p in yolo.parameters()))
    print(out.shape)
    print(out[..., :10])
    # print(yolo)