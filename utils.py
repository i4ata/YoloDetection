import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import nms, box_convert

from typing import List

GRID = torch.stack(torch.meshgrid(torch.arange(7), torch.arange(7), indexing='ij'), dim=1)

def iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    # [x,y,w,h]
    b1x1 = box1[..., 0] - box1[..., 2] / 2
    b1x2 = box1[..., 0] + box1[..., 2] / 2
    b1y1 = box1[..., 1] - box1[..., 3] / 2
    b1y2 = box1[..., 1] + box1[..., 3] / 2
    
    b2x1 = box2[..., 0] - box2[..., 2] / 2
    b2x2 = box2[..., 0] + box2[..., 2] / 2
    b2y1 = box2[..., 1] - box2[..., 3] / 2
    b2y2 = box2[..., 1] + box2[..., 3] / 2
    
    x1 = torch.max(b1x1, b2x1) 
    y1 = torch.max(b1y1, b2y1) 
    x2 = torch.min(b1x2, b2x2) 
    y2 = torch.min(b1y2, b2y2) 
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) 
    union = box1[..., 2] * box1[..., 3] + box2[..., 2] * box2[..., 3] - intersection

    return intersection / (union + 1e-6)

class YOLOv1Loss(nn.Module):
    def __init__(self, n_boxes: int = 2, n_classes: int = 20, lambda_coord: float = 5., lambda_noobj: float = .5) -> None:
        super(YOLOv1Loss, self).__init__()
        
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.n_boxes = n_boxes
        self.n_classes = n_classes
        
        self.boxes_dims = n_boxes * 5
        self.output_dims = self.boxes_dims + n_classes

        
    def forward(self, y_pred: torch.Tensor, target_boxes: List[torch.Tensor], target_labels: List[torch.Tensor]) -> torch.Tensor:

        # Transform targets to yolo coordinates
        y_true = torch.stack(
            [transform_to_yolo(boxes=boxes, labels=labels) for boxes, labels in zip(target_boxes, target_labels)]
        )

        # Extract the bounding boxes [batch_size, S, S, B, 5]
        boxes = y_pred[..., :self.boxes_dims].view(*y_pred.shape[:-1], self.n_boxes, 5)

        # Compute ious [batch_size, S, S, B]
        ious = torch.stack([iou(box1=boxes[..., i, :], box2=y_true[..., :4]) for i in range(self.n_boxes)], dim=-1)
        
        # Compute the true bounding boxes (the ones with the highest iou) [batch_size, S, S, 5]
        true_boxes = torch.gather(
            input=boxes, 
            dim=3, 
            index=ious.argmax(-1, keepdim=True).unsqueeze(-1).expand(-1,-1,-1,-1,5)
        ).squeeze(3)

        # Get the 2 masks
        obj_mask = y_true[..., 4] != 0
        noobj_mask = ~obj_mask

        return (
            self.lambda_coord * (
                F.mse_loss(true_boxes[..., 0][obj_mask], y_true[..., 0][obj_mask]) +
                F.mse_loss(true_boxes[..., 1][obj_mask], y_true[..., 1][obj_mask])
            ) + 
            self.lambda_coord * (
                F.mse_loss(true_boxes[..., 2][obj_mask].sqrt(), y_true[..., 2][obj_mask].sqrt()) +
                F.mse_loss(true_boxes[..., 3][obj_mask].sqrt(), y_true[..., 2][obj_mask].sqrt())
            ) + 
            F.binary_cross_entropy(true_boxes[..., 4][obj_mask], y_true[..., 4][obj_mask]) + 
            self.lambda_noobj * (
                F.binary_cross_entropy(true_boxes[..., 4][noobj_mask], y_true[..., 4][noobj_mask])
            ) +
            F.cross_entropy(y_pred[..., self.boxes_dims:][obj_mask], y_true[..., 5][obj_mask].long())
        )

def transform_to_yolo(boxes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

    x, y, w, h = box_convert(boxes=boxes, in_fmt='xyxy', out_fmt='cxcywh').T
    x, y = x / 64, y / 64
    w, h = w / 448, h / 448

    objectness_score = torch.ones(len(boxes), device=boxes.device)

    feature_map = torch.zeros(7, 7, 6)
    feature_map[x.long(), y.long()] = torch.stack(
        (x.frac(), y.frac(), w, h, objectness_score, labels), dim=1
    )

    return feature_map

def transform_outputs(detections: torch.Tensor, confidence_threshold: float = .3, iou_threshold: float = .3) -> List[torch.Tensor]:
    detections = [x[x[:, 4] < confidence_threshold] for x in detections]
    detections = [
        x[nms(
            boxes=box_convert(boxes=x[:, :4], in_fmt='cxcywh', out_fmt='xyxy'), 
            scores=x[:, 4], 
            iou_threshold=iou_threshold
        )]
        for x in detections
    ]
    return detections

def transform_from_yolo(detections: torch.Tensor) -> torch.Tensor:
    detections[..., [0,1]] *= 64 * GRID
    detections[..., [0,2]] *= 448
    return detections.flatten(start_dim=1, end_dim=2)

if __name__ == '__main__':
    torch.manual_seed(0)
    l = YOLOv1Loss()
    yt, yp = torch.rand(3,7,7,6), torch.rand(3,7,7,30)
    yt[:,:,:,4] = yt[:,:,:,4] > .5
    loss = l(yp,yt)
    print(loss)