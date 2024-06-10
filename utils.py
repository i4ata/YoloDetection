import torch
from torchvision.ops import box_convert, box_iou, nms
from functools import partial

from typing import List, Literal, Dict, List

S = 7
IMAGE_SIZE = 448
STRIDE = IMAGE_SIZE / S
GRID = torch.stack(torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij'), dim=-1)
_convert_fun_3 = partial(torch.vmap(torch.vmap(torch.vmap(box_convert))), in_fmt='cxcywh', out_fmt='xyxy')
_iou_fun_3 = torch.vmap(torch.vmap(torch.vmap(box_iou)))
_convert_fun_1 = partial(torch.vmap(box_convert), in_fmt='cxcywh', out_fmt='xyxy')


def transform_from_yolo(
        feature_map: torch.Tensor, 
        objectness_threshold: float = .5, 
        iou_threshold: float = .3
    ) -> List[Dict[Literal['boxes', 'scores', 'labels'], torch.Tensor]]:
    """
    Input: Batched detections [batch size, S, S, [x,y,w,h,c,C]]
    """
    feature_map[..., [0,1]] += GRID.to(feature_map.device)
    feature_map[..., [0,1]] *= STRIDE
    feature_map[..., [2,3]] *= IMAGE_SIZE

    detections = feature_map.flatten(start_dim=1, end_dim=2)
    detections = torch.cat((detections[..., :5], detections[..., 5:].argmax(dim=-1, keepdim=True)), dim=-1)
    detections[..., :4] = _convert_fun_1(detections[..., :4])
    detections = [x[x[:, 4] > objectness_threshold] for x in detections]
    detections = [x[nms(boxes=x[:, :4], scores=x[:, 4], iou_threshold=iou_threshold)] for x in detections]
    output = [
        {
            'boxes': x[:, :4],
            'scores': x[:, 4],
            'labels': x[:, 5].long()
        } 
        for x in detections
    ]
    return output

def transform_to_yolo(boxes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Input: 
    - Bounding boxes [N, [xyxy]]
    - Labels [N, 1]

    Output: The corresponding feature map [S, S, [x,y,w,h,1,C]] 
    """
    boxes = box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')

    x, y, w, h = boxes.T
    x, y = x / STRIDE, y / STRIDE
    w, h = w / IMAGE_SIZE, h / IMAGE_SIZE

    objectness_score = torch.ones(len(boxes), device=boxes.device)

    feature_map = torch.zeros(S, S, 6, device=boxes.device)
    feature_map[x.long(), y.long()] = torch.stack(
        (x.frac(), y.frac(), w, h, objectness_score, labels), dim=1
    )

    return feature_map

def iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Input: 
    - pred_boxes: predicted boxes [batch size, S, S, B, [xywh]]
    - target_boxes: the ground truth boxes [batch size, S, S, [xywh]]
    Output:
    - iou_scores: [batch size, S, S, B]
    """

    pred_boxes = _convert_fun_3(pred_boxes)
    target_boxes = _convert_fun_3(target_boxes.unsqueeze(-2))
    
    iou_scores = _iou_fun_3(pred_boxes, target_boxes).squeeze(-1)
    
    return iou_scores
