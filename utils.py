import torch

from typing import List, Tuple, Literal, Union

DEBUG = True

def predict_transform(prediction: torch.Tensor, input_dim: int, anchors: List[Tuple[int, int]], num_classes: int) -> torch.Tensor:
    
    if DEBUG: 
        print(f'*' * 50)
        print('In transformation of the model outputs. Goal: transform the model outputs to a meaningful format that is coherent with the original image')
        print(f'Inputs: prediction -> {prediction.shape}, input image dim -> {input_dim}, anchors -> {anchors}, number of classes: {num_classes}')

    if DEBUG: print(f'Definitions:')
    if DEBUG: print(f'Prediction shape: [batch size, bbox attributes * num anchors, feature map height, feature map width] -> {prediction.shape}')
    batch_size = len(prediction)
    stride = input_dim // prediction.shape[2] # The factor by which the input image is reduced
    if DEBUG: print(f'Stride: factor by which the input image is reduced: input image dim // feature map height = {stride}')
    grid_size = input_dim // stride # Size of the grid represented by a cell in the output feature map
    if DEBUG: print(f'The size of a region in the original image represented by a single cell in the output feature map: Input image dim // stride = {grid_size}')
    bbox_attributes = 5 + num_classes # x, y, w, h (4), objectness score (1), confidence for each class (num_classes)
    if DEBUG: print(f'The number of attributes per 1 bbox: coordinates of its center (x,y) + dimensions (h,w) + objectness score (c)  + class score (num_classes) = {bbox_attributes}') 
    num_anchors = len(anchors)
    if DEBUG: print(f'Number of anchors: {num_anchors}')

    prediction = prediction.view(batch_size, bbox_attributes * num_anchors, grid_size ** 2)
    if DEBUG: print(f'1. Flatten the last 2 dimensions of the feature map: Prediction -> {prediction.shape}')

    prediction = prediction.transpose(1,2).contiguous()
    if DEBUG: print(f'2. Swap feature map dimensions with the bboxes and anchors: Prediction -> {prediction.shape}')

    prediction = prediction.view(batch_size, num_anchors * grid_size ** 2, bbox_attributes)
    if DEBUG: print(f'3. Push the anchors to the feature map. That is, create (num anchors)-many feature maps: Prediction -> {prediction.shape}')

    anchors = [(a1 / stride, a2 / stride) for a1, a2 in anchors]
    if DEBUG: print(f'4. Normalize the anchors so that they are coherent with the feature map: {anchors}')

    prediction[:, :, [0,1,4]] = torch.sigmoid(prediction[:,:,[0,1,4]])
    if DEBUG: 
        print(f'5. Normalize the xs and ys of the boxes as well as the objectness score using sigmoid')
        print(f'That is done since the xs and ys represent proportional offsets with respect to the anchors, not coordinates')

    # Add the offsets to x and y
    if DEBUG: print(f'6. Add the offsets from the top left of the feature map to x and y according to the formula')
    grid = torch.arange(grid_size, device=prediction.device)
    a, b = torch.meshgrid(grid, grid)
    x_offset, y_offset = a.reshape(-1, 1), b.reshape(-1, 1)
    x_y_offset = torch.cat((x_offset, y_offset), dim=1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset

    # Transform w and h
    if DEBUG: print(f'7. Do the logarithmic transform of w and h and multiply by the anchors dimensions according to the formula')
    anchors = torch.tensor(anchors, device=prediction.device).repeat(grid_size ** 2, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Normalize class scores
    if DEBUG: print(f'8. Normalize the class confidences by taking the sigmoid')
    prediction[:, :, 5:5+num_classes] = torch.sigmoid(prediction[:, :, 5:5+num_classes])

    # Resize bounding boxes to the size of the image
    if DEBUG: print(f'9. Scale x,y,w,h with the stride to obtain the dimensions with respect to the original image instead of the feature map')
    prediction[:, :, :4] *= stride

    if DEBUG: print(f'!Return predictions [batch_size, num_anchors * feature_map_i, bbox_attributes] -> {prediction.shape}!')
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
    
    if DEBUG: 
        print('In write results. Goals: extract the true detections from all detections using thresholding and nms')
        print(f'Inputs: predictions [batch_size, num_anchors * feature_map_id, bbox] -> {prediction.shape}, objectness score threshold = {confidence}, num classes = {num_classes}, nms confidence = {nms_conf}')
    
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction *= conf_mask
    if DEBUG: print(f'1. zero-out all predictions that have a lower objectness score than the threshold')

    # x_center, y_center, height, width -> x_topleft, y_topleft, x_bottomright, y_bottomright
    box_corner = torch.clone(prediction)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]) / 2
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]) / 2
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]) / 2
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]) / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    if DEBUG: print(f'2. Transform the bbox format from [x_center, y_center, height, width] to [x_topleft, y_topleft, x_bottomright, y_bottomright]')

    batch_size = len(prediction)
    write = False

    if DEBUG: print(f'Loop over each image in the batch')
    for ind in range(batch_size):
        image_pred = prediction[ind]

        max_conf, max_conf_score = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        image_pred = torch.cat((image_pred[:, :5], max_conf, max_conf_score), 1)
        if DEBUG and not ind: 
            print(f'3. Since yolov3 detects 1 object per cell in the feature map, get the object with the class confidence, and its confidence')
            print(f'So the shape of the bbox is transformed from [x1 + y1 + x2 + y1 + objectness score + classes] to [x1 + y1 + x2 + y2 + objectness score + top class index + top class confidence]')

        # torch.nonzero(x) returns the indices where x != 0
        non_zero_ind = torch.nonzero(image_pred[:, 4])

        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue
        if DEBUG and not ind: print(f'3.1 Get only the cells of the image that predict an object (using the prediction mask). Thresholding done')
        

        img_classes: torch.Tensor = torch.unique(image_pred_[:, -1])
        if DEBUG and not ind: 
            print(f'4. Get the unique classes that are predicted in that image')
            print(f'Now loop over all unique classes')

        for cls in img_classes:
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            if DEBUG and not ind and img_classes[0] == cls:
                print(f'4.1 Get only the boxes that detect that class')

            conf_sort_index = torch.argsort(image_pred_class[:, 4], descending=True)
            image_pred_class = image_pred_class[conf_sort_index]
            idx = len(image_pred_class) # number of detections
            if DEBUG and not ind and img_classes[0] == cls:
                print(f'4.2 Order the boxes by objectness score in descending order')
                print(f'Loop over them')
                print(f'For each box remove all boxes after it that have a IOU with the current one higher than the threshold')
                print(f'That way overlapping boxes are removed')

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

            batch_ind = torch.full(size=(len(image_pred_class), 1), fill_value=ind, device=prediction.device)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    if DEBUG: 
        print(f'Returned: Cleaned (true) detections: [batch_size, n, (x1 + x2 + y1 + y2 + Objectness score + Predicted class + Confidence)]')
        print(f'Here, n is the length of the true predictions, if there are any')
    try:
        return output
    except:
        return 0