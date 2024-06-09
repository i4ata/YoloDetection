import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import iou

class YOLOv1Loss(nn.Module):
    def __init__(self, n_boxes: int = 2, n_classes: int = 20, lambda_coord: float = 5., lambda_noobj: float = .5) -> None:
        super(YOLOv1Loss, self).__init__()
        
        self.n_boxes = n_boxes
        self.n_classes = n_classes
        
        self.boxes_dims = n_boxes * 5
        self.output_dims = self.boxes_dims + n_classes

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred shape: [batch_size, 7, 7, (B * [cx,cy,w,h,c]) + C]
        y_true_shape: [batch_size, 7, 7, [cx,cy,w,h,c,i]], where i is the index of the class
        """

        obj_mask = y_true[..., 4] != 0
        noobj_mask = ~obj_mask

        # Extract the bounding boxes [batch_size, S, S, B, 5]
        pred_boxes = y_pred[..., :self.boxes_dims].view(*y_pred.shape[:-1], self.n_boxes, 5)

        # For each cell, get the ious between the predicted bounding boxes and the true one [batch_size, S, S, B]
        ious = iou(pred_boxes[..., :4], y_true[..., :4])

        # For each cell, get the largest iou and the largest index ([batch_size, S, S], [batch_size, S, S])
        max_ious, best_ious = ious.max(dim=-1)

        # Compute the true bounding boxes (the ones with the highest iou) [batch_size, S, S, 5]
        detections = torch.gather(
            input=pred_boxes, 
            dim=3, 
            index=best_ious.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,-1,5)
        ).squeeze(-2)

        # The model predicts the IOU
        y_true[..., 4] *= max_ious

        detections = torch.cat((detections, y_pred[..., self.boxes_dims:]), dim=-1)

        loss = (
            # x
            self.lambda_coord * F.mse_loss(detections[..., 0][obj_mask], y_true[..., 0][obj_mask]) + 
            # y
            self.lambda_coord * F.mse_loss(detections[..., 1][obj_mask], y_true[..., 1][obj_mask]) + 
            # w
            self.lambda_coord * F.mse_loss(detections[..., 2][obj_mask].sqrt(), y_true[..., 2][obj_mask].sqrt()) + 
            # h
            self.lambda_coord * F.mse_loss(detections[..., 3][obj_mask].sqrt(), y_true[..., 3][obj_mask].sqrt()) + 
            # c_obj
            F.mse_loss(detections[..., 4][obj_mask], y_true[..., 4][obj_mask]) + 
            # c_noobj
            self.lambda_noobj * F.mse_loss(detections[..., 4][noobj_mask], y_true[..., 4][noobj_mask]) + 
            # C
            F.mse_loss(detections[..., 5:][obj_mask], F.one_hot(y_true[..., 5][obj_mask].long(), num_classes=self.n_classes).float())
        )

        return loss