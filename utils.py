import torch
import torch.nn as nn
def iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    # [x,y,w,h]
    b1x1 = box1[:, 0] - box1[:, 2] / 2
    b1x2 = box1[:, 0] + box1[:, 2] / 2
    b1y1 = box1[:, 1] - box1[:, 3] / 2
    b1y2 = box1[:, 1] + box1[:, 3] / 2
    
    b2x1 = box2[:, 0] - box2[:, 2] / 2
    b2x2 = box2[:, 0] + box2[:, 2] / 2
    b2y1 = box2[:, 1] - box2[:, 3] / 2
    b2y2 = box2[:, 1] + box2[:, 3] / 2
    
    x1 = torch.max(b1x1, b2x1) 
    y1 = torch.max(b1y1, b2y1) 
    x2 = torch.min(b1x2, b2x2) 
    y2 = torch.min(b1y2, b2y2) 
     
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) 
    union = box1[:, 2] * box1[:, 3] + box2[:, 2] * box2[:, 3] - intersection

    return intersection / (union + 1e-6)

class YOLOv1Loss(nn.Module):
    def __init__(self) -> None:
        super(YOLOv1Loss, self).__init__()
        
        self.lambda_coord = 5
        self.lambda_noobj = .5
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # box_i: [n,7,7,5+C]
        box1, box2 = y_pred[:, :, :, :5], y_pred[:, :, :, 5:10]
        true_boxes = torch.zeros_like(box1)
        obj_mask, noobj_mask = y_true[:, :, :, 4] != 0, y_true[:, :, :, 4] == 0
        
        box1_iou = iou(box1=box1[obj_mask], box2=true_boxes[obj_mask][:, :4])
        box2_iou = iou(box1=box2[obj_mask], box2=true_boxes[obj_mask][:, :4])
        iou_mask = box1_iou >= box2_iou
        true_boxes[obj_mask][iou_mask] = box1[obj_mask][iou_mask]
        true_boxes[obj_mask][~iou_mask] = box2[obj_mask][~iou_mask]
        #print(true_boxes.shape)
        #a
        #for i in range(len(true_boxes[obj_mask])):
        #    true_boxes[obj_mask][i] = box1[obj_mask][i] if box1_iou[i] >= box2_iou[i] else box2[obj_mask][i]
        #a
        #true_boxes[obj_mask] = torch.where(box1_iou >= box2_iou, box1[obj_mask], box2[obj_mask])
        
        # for i, image in enumerate(y_true):
        #     for x, image in enumerate(image):
        #         for y, image in enumerate(image):
        #             if obj_mask[i,x,y]:
        #                 box1_iou = iou(box1=box1[i,x,y], box2=image)
        #                 box2_iou = iou(box1=box2[i,x,y], box2=image)
        #                 true_boxes[i,x,y] = (box1 if box1_iou > box2_iou else box2)[i,x,y]
        x_loss = self.mse_loss(true_boxes[:, :, :, 0], y_true[:, :, :, 0])
        y_loss = self.mse_loss(true_boxes[:, :, :, 1], y_true[:, :, :, 1])
        w_loss = self.mse_loss(torch.sqrt(true_boxes[:, :, :, 2]), torch.sqrt(y_true[:, :, :, 2]))
        h_loss = self.mse_loss(torch.sqrt(true_boxes[:, :, :, 3]), torch.sqrt(y_true[:, :, :, 3]))
        obj_loss = self.bce_loss(true_boxes[obj_mask][:, 4], y_true[obj_mask][:, 4])
        noobj_loss = self.bce_loss(true_boxes[noobj_mask][:, 4], y_true[noobj_mask][:, 4])
        class_loss = self.cross_entropy_loss(y_pred[obj_mask][:, 10:], y_true[obj_mask][:, 5].long())

        #print(x_loss, y_loss, w_loss, h_loss, obj_loss, noobj_loss, class_loss)

        return (
            self.lambda_coord * (x_loss + y_loss) +
            self.lambda_coord * (w_loss + h_loss) +
            obj_loss +
            self.lambda_noobj * noobj_loss +
            class_loss
        )
        

if __name__ == '__main__':
    a = torch.rand(3,7,7,5)
    m = a[:,:,:,4] > .5
    b1, b2 = torch.rand_like(a), torch.rand_like(a)
    m2 = b1[m].sum(1) > b2[m].sum(1)
    print(a.shape, m.shape, m2.shape, a[m].shape)
    a[m] = torch.where(m2, b1[m], b2[m])