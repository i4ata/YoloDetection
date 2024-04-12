import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_convert

from typing import List

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
            self.lambda_coord * (
                F.mse_loss(y_pred[..., 0][obj_mask], y_true[..., 0][obj_mask]) +
                F.mse_loss(y_pred[..., 1][obj_mask], y_true[..., 1][obj_mask])
            ) + 
            self.lambda_coord * (
                F.mse_loss(y_pred[..., 2][obj_mask].sqrt(), y_true[..., 2][obj_mask].sqrt()) +
                F.mse_loss(y_pred[..., 3][obj_mask].sqrt(), y_true[..., 2][obj_mask].sqrt())
            ) + 
            F.binary_cross_entropy(y_pred[..., 4][obj_mask], y_true[..., 4][obj_mask]) + 
            self.lambda_noobj * (
                F.binary_cross_entropy(y_pred[..., 4][noobj_mask], y_true[..., 4][noobj_mask])
            ) +
            F.cross_entropy(y_pred[..., 5:][obj_mask], y_true[..., 5][obj_mask].long())
        )

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

if __name__ == '__main__':
    torch.manual_seed(0)
    l = YOLOv1Loss()
    yt, yp = torch.rand(3,7,7,6), torch.rand(3,7,7,30)
    yt[:,:,:,4] = yt[:,:,:,4] > .5
    loss = l(yp,yt)
    print(loss)