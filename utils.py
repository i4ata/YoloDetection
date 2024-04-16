import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_convert

from typing import List

# This is a [7,7,2] grid, where the [i,j]-th cell is equal to [i,j]
GRID = torch.stack(torch.meshgrid(torch.arange(7), torch.arange(7), indexing='ij'), dim=-1)

class YOLOv1Loss(nn.Module):
    def __init__(self, lambda_coord: float = 5., lambda_noobj: float = .5) -> None:
        super(YOLOv1Loss, self).__init__()
        
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred shape: [batch_size, 7, 7, [cx,cy,w,h,c] + C]
        y_true_shape: [batch_size, 7, 7, [cx,cy,w,h,1,i]], where i is the index of the class
        """

        # Get the 2 masks
        obj_mask = y_true[..., 4] != 0
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

def postprocess_outputs(detections: torch.Tensor, confidence_threshold: float = .5, iou_threshold: float = .3) -> List[torch.Tensor]:
    detections = [x[x[:, 4] > confidence_threshold] for x in detections]
    detections = [
        x[nms(
            boxes=box_convert(boxes=x[:, :4], in_fmt='cxcywh', out_fmt='xyxy'), 
            scores=x[:, 4], 
            iou_threshold=iou_threshold
        )] 
        for x in detections
    ]
    return detections

def transform_to_yolo(boxes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

    x, y, w, h = boxes.T
    x, y = x / 64., y / 64.
    w, h = w / 448., h / 448.

    objectness_score = torch.ones(len(boxes), device=boxes.device)

    feature_map = torch.zeros(7, 7, 6, device=boxes.device)
    feature_map[x.long(), y.long()] = torch.stack(
        (x.frac(), y.frac(), w, h, objectness_score, labels), dim=1
    )

    return feature_map

def transform_from_yolo(detections: torch.Tensor) -> torch.Tensor:
    detections[..., [0,1]] = (detections[..., [0,1]] + GRID.to(detections.device)) * 64
    detections[..., [2,3]] *= 448
    return detections.flatten(start_dim=1, end_dim=2)

if __name__ == '__main__':
    torch.manual_seed(0)
    l = YOLOv1Loss()
    yt, yp = torch.rand(3,7,7,6), torch.rand(3,7,7,30)
    yt[:,:,:,4] = yt[:,:,:,4] > .5
    loss = l(yp,yt)
    print(loss)