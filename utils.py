import torch
from torchvision.ops import box_convert, box_iou
from functools import partial

S = 7
IMAGE_SIZE = 448
STRIDE = IMAGE_SIZE / S
GRID = torch.stack(torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij'), dim=-1)

def transform_from_yolo(x: torch.Tensor) -> torch.Tensor:
    """
    Input: Batched detections [batch size, S, S, [x,y,w,h,c,C]]
    Output: Bounding boxes normalized to image coordinates [batch size, S^2, [xy,w,h,c,C]]
    """
    x[..., [0,1]] += GRID.to(x.device)
    x[..., [0,1]] *= STRIDE
    x[..., [2,3]] *= IMAGE_SIZE
    return x.flatten(start_dim=1, end_dim=2)

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

_convert_fun = partial(torch.vmap(torch.vmap(torch.vmap(box_convert))), in_fmt='cxcywh', out_fmt='xyxy')
_iou_fun = torch.vmap(torch.vmap(torch.vmap(box_iou)))

def iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Input: 
    - pred_boxes: predicted boxes [batch size, S, S, B, [xywh]]
    - target_boxes: the ground truth boxes [batch size, S, S, [xywh]]
    Output:
    - iou_scores: [batch size, S, S, B]
    """

    pred_boxes = _convert_fun(pred_boxes)
    target_boxes = _convert_fun(target_boxes.unsqueeze(-2))
    
    iou_scores = _iou_fun(pred_boxes, target_boxes).squeeze(-1)
    
    return iou_scores


if __name__ == '__main__':

    b1 = torch.rand(3,7,7,2,4)
    b2 = torch.rand(3,7,7,4)
    out = iou(b1, b2)
    print(out.isnan().any())
    print(out.shape)
    a,b = out.max(-1)
    print(a, b)