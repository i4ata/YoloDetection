import torch

from typing import List, Tuple, Literal, Union

def predict_transform(prediction: torch.Tensor, input_dim: int, anchors: List[Tuple[int, int]], num_classes: int) -> torch.Tensor:
    batch_size = len(prediction)
    stride = input_dim // prediction.shape[2] # The factor by which the input image is reduced
    grid_size = input_dim // stride # Size of the grid represented by a cell in the output
    bbox_attributes = 5 + num_classes # x, y, w, h, objectness score, confidence for each class 
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attributes * num_anchors, grid_size ** 2)
    prediction = prediction.transpose(1,2).contiguous()
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

def bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.max(b1_x2, b2_x2)
    inter_rect_y2 = torch.max(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou
    
def write_results(prediction: torch.Tensor, confidence: float, num_classes: int, nms_conf: float = .4) -> Union[torch.Tensor, Literal[0]]:
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction *= conf_mask

    # x_center, y_center, height, width -> x_topleft, y_topleft, x_bottomright, y_bottomright
    box_corner = torch.clone(prediction)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]) / 2
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]) / 2
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]) / 2
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]) / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    
    batch_size = len(prediction)
    write = False

    for i in range(batch_size):
        image_pred = prediction[i]

        max_conf, max_conf_score = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        image_pred = torch.cat((image_pred[:, :5], max_conf, max_conf_score), 1)

        non_zero_ind = torch.nonzero(image_pred[:, 4])

        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue

        img_classes: torch.Tensor = torch.unique(image_pred_[:, -1])

        for cls in img_classes:
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = len(image_pred_class) # number of detections

            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = torch.full(size=(len(image_pred_class), 1), fill_value=i, device=prediction.device)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    try:
        return output
    except:
        return 0