# https://github.com/pjreddie/darknet/blob/master/cfg/yolov1-tiny.cfg

import torch
import torch.nn as nn
from torchvision.ops import box_iou, box_convert
from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from utils import YOLOv1Loss, transform_from_yolo, transform_to_yolo, postprocess_outputs

from typing import Tuple, List

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=.1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
class FastYOLO1(LightningModule):
    def __init__(self, n_boxes: int = 2, n_classes: int = 20) -> None:
        super(FastYOLO1, self).__init__()
        
        self.n_boxes = n_boxes
        self.n_classes = n_classes

        self.boxes_dims = n_boxes * 5
        self.output_dims = self.boxes_dims + n_classes

        pool = nn.MaxPool2d(2)
        self.net = nn.Sequential(
            ConvBlock(3, 16), pool, 
            ConvBlock(16, 32), pool,
            ConvBlock(32, 64), pool,
            ConvBlock(64, 128), pool,
            ConvBlock(128, 256), pool,
            ConvBlock(256, 512), pool,
            ConvBlock(512, 1024),
            ConvBlock(1024, 256),
            nn.Flatten(),
            nn.Linear(in_features=256 * 7 * 7, out_features= 7 * 7 * self.output_dims)
        )

        self.loss_fn = YOLOv1Loss()
        # self.map = MeanAveragePrecision(box_format='cxcywh')
        self.iou_func = torch.vmap(torch.vmap(torch.vmap(box_iou)))
        self.box_convert_func = torch.vmap(torch.vmap(torch.vmap(box_convert)))

        self.losses = {'train': [], 'val': []}
        self.maps = {'train': MeanAveragePrecision(box_format='cxcywh'), 'val': MeanAveragePrecision(box_format='cxcywh')}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detections: torch.Tensor = self.net(x)
        detections = detections.view(len(x), 7, 7, self.output_dims)
        detections[..., :self.boxes_dims] = detections[..., :self.boxes_dims].sigmoid()
        detections[..., self.boxes_dims:] = detections[..., self.boxes_dims:].softmax(-1)
        
        return detections
    
    def _process_output(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # Extract the bounding boxes [batch_size, S, S, B, 5]
        boxes = y_pred[..., :self.boxes_dims].view(*y_pred.shape[:-1], self.n_boxes, 5)

        ious: torch.Tensor = self.iou_func(
            self.box_convert_func(boxes[..., :4], in_fmt='cxcywh', out_fmt='xyxy'),
            self.box_convert_func(y_true[..., :4].unsqueeze(-2), in_fmt='cxcywh', out_fmt='xyxy')
        ).squeeze(-1)

        # Compute the true bounding boxes (the ones with the highest iou) [batch_size, S, S, 5]
        detections = torch.gather(
            input=boxes, 
            dim=3, 
            index=ious.argmax(-1, keepdim=True).unsqueeze(-1).expand(-1,-1,-1,-1,5)
        ).squeeze(-2)

        detections = torch.cat((detections, y_pred[..., self.boxes_dims:]), dim=-1)
        return detections

    def _get_map(self, detections: torch.Tensor, target_boxes: List[torch.Tensor], target_labels: List[torch.Tensor], stage: str = 'train'):

        detections = transform_from_yolo(detections=detections.detach())
        detections = postprocess_outputs(detections=detections)

        preds = [{
            'boxes': x[..., :4],
            'scores': x[..., 4],
            'labels': x[..., 5:].argmax(-1)
        } for x in detections]
        
        targets = [{
            'boxes': boxes,
            'labels': labels
        } for boxes, labels in zip(target_boxes, target_labels)]

        self.maps[stage].update(preds=preds, target=targets)

    def training_step(self, batch: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:

        inputs, target_boxes, target_labels = batch
        y_pred: torch.Tensor = self(inputs)

        # Transform targets to yolo coordinates
        y_true = torch.stack(
            [transform_to_yolo(boxes=boxes, labels=labels) for boxes, labels in zip(target_boxes, target_labels)]
        )

        detections = self._process_output(y_pred=y_pred, y_true=y_true)

        loss = self.loss_fn(detections, y_true)
        self.losses['train'].append(loss)
        
        self._get_map(detections=detections, target_boxes=target_boxes, target_labels=target_labels)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        inputs, target_boxes, target_labels = batch
        y_pred: torch.Tensor = self(inputs)

        # Transform targets to yolo coordinates
        y_true = torch.stack(
            [transform_to_yolo(boxes=boxes, labels=labels) for boxes, labels in zip(target_boxes, target_labels)]
        )

        detections = self._process_output(y_pred=y_pred, y_true=y_true)

        loss = self.loss_fn(detections, y_true)
        self.losses['val'].append(loss)
        self._get_map(detections=detections, target_boxes=target_boxes, target_labels=target_labels, stage='val')

        return loss

    def _log(self, stage: str = 'train') -> None:
        self.log_dict({
            f'{stage}_loss': torch.stack(self.losses[stage]).mean(), 
            f'{stage}_map': self.maps[stage].compute()['map']}, 
        prog_bar=True)
        self.losses[stage].clear()
        self.maps[stage].reset()

    def on_train_epoch_end(self) -> None:
        self._log('train')

    def on_validation_epoch_end(self) -> None:
        self._log('val')
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(params=self.net.parameters(), lr=.001)

if __name__ == '__main__':
    yolo = FastYOLO1()
    im = torch.rand(5,3,448,448)
    out = yolo(im)
    print(sum(p.numel() for p in yolo.parameters()))
    print(out.shape)
    print(out[..., :10])
    # print(yolo)