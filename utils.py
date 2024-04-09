import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self) -> None:
        super(YOLOv1Loss, self).__init__()
        
        self.lambda_coord = 5
        self.lambda_noobj = .5
        
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:

        # Extract the two bounding boxes per each cell [x,y,w,h,c]
        box1, box2 = y_pred[..., :5], y_pred[..., 5:10]

        # Get the 2 masks
        obj_mask = y_true[..., 4] != 0
        noobj_mask = ~obj_mask

        # Compute iou between the predicted boxes and true ones
        box1_iou = iou(box1=box1, box2=y_true[..., :4]) 
        box2_iou = iou(box1=box2, box2=y_true[..., :4])

        # Compute the true bounding boxes (the true box is the one with the higher iou)
        true_boxes = torch.where(
            condition=(
                box1_iou > box2_iou
            ).unsqueeze(-1),
            input=box1, 
            other=box2
        )

        # Transform the confidence scores
        # true_boxes[:,:,:,4][obj_mask] *= torch.max(box1_iou, box2_iou) # P(Object) * IOU
        
        return (
            self.lambda_coord * (
                F.mse_loss(true_boxes[..., 0][obj_mask], y_true[..., 0][obj_mask]) +
                F.mse_loss(true_boxes[..., 1][obj_mask], y_true[..., 1][obj_mask])
            ) + 
            self.lambda_coord * (
                F.mse_loss(torch.sqrt(true_boxes[..., 2][obj_mask]), torch.sqrt(y_true[..., 2][obj_mask])) +
                F.mse_loss(torch.sqrt(true_boxes[..., 3][obj_mask]), torch.sqrt(y_true[..., 2][obj_mask]))
            ) + 
            F.binary_cross_entropy_with_logits(true_boxes[..., 4][obj_mask], y_true[..., 4][obj_mask]) + 
            self.lambda_noobj * (
                F.binary_cross_entropy_with_logits(true_boxes[..., 4][noobj_mask], y_true[..., 4][noobj_mask])
            ) +
            F.cross_entropy(y_pred[..., 10:][obj_mask], y_true[..., 5][obj_mask].long())
        )
        

if __name__ == '__main__':
    torch.manual_seed(0)
    l = YOLOv1Loss()
    yt, yp = torch.rand(3,7,7,6), torch.rand(3,7,7,30)
    yt[:,:,:,4] = yt[:,:,:,4] > .5
    loss = l(yt,yp)
    print(loss)