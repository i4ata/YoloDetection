# https://github.com/pjreddie/darknet/blob/master/cfg/yolov1-tiny.cfg

import torch
import torch.nn as nn
from torchvision.ops import box_iou, box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image, resize
from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision
from typing import Tuple, List, Literal
from PIL import Image

from utils import YOLOv1Loss, transform_from_yolo, transform_to_yolo, postprocess_outputs


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
    
    def _process_output(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Extract the bounding boxes [batch_size, S, S, B, 5]
        boxes = y_pred[..., :self.boxes_dims].view(*y_pred.shape[:-1], self.n_boxes, 5)

        # Get the obj_mask
        obj_mask = y_true[..., 4] != 0

        # For each cell, get the ious between the predicted bounding boxes and the true one [batch_size, S, S, 5]
        ious: torch.Tensor = self.iou_func(
            self.box_convert_func(boxes[..., :4], in_fmt='cxcywh', out_fmt='xyxy'),
            self.box_convert_func(y_true[..., :4].unsqueeze(-2), in_fmt='cxcywh', out_fmt='xyxy')
        ).squeeze(-1)

        # For each cell, get the largest iou and the largest index ([batch_size, S, S], [batch_size, S, S])
        max_ious, best_ious = ious.max(dim=-1)

        # Compute the true bounding boxes (the ones with the highest iou) [batch_size, S, S, 5]
        detections = torch.gather(
            input=boxes, 
            dim=3, 
            index=best_ious.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,-1,5)
        ).squeeze(-2)

        # The model predicts the IOU
        y_true[..., 4] *= max_ious

        detections = torch.cat((detections, y_pred[..., self.boxes_dims:]), dim=-1)
        return detections, y_true, obj_mask

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

    def _step(self, batch:  Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]], stage: Literal['train', 'val']) -> torch.Tensor:
        inputs, target_boxes, target_labels = batch
        y_pred: torch.Tensor = self(inputs)

        # Transform targets to yolo coordinates
        y_true = torch.stack(
            [transform_to_yolo(boxes=boxes, labels=labels) for boxes, labels in zip(target_boxes, target_labels)]
        )

        detections, y_true, obj_mask = self._process_output(y_pred=y_pred, y_true=y_true)

        loss = self.loss_fn(detections, y_true, obj_mask)
        self.losses[stage].append(loss)
        
        self._get_map(detections=detections, target_boxes=target_boxes, target_labels=target_labels, stage=stage)
        
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self._step(batch=batch, stage='train')

    def validation_step(self, batch: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self._step(batch=batch, stage='val')

    def _log(self, stage: Literal['train', 'val']) -> None:
        self.log_dict({
            f'{stage}_loss': torch.stack(self.losses[stage]).mean(), 
            f'{stage}_map': self.maps[stage].compute()['map']}, 
        prog_bar=True)
        self.losses[stage].clear()
        self.maps[stage].reset()

    def on_train_epoch_end(self) -> None:
        self._log(stage='train')

    def on_validation_epoch_end(self) -> None:
        self._log(stage='val')
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(params=self.net.parameters(), lr=1e-3, momentum=.9, weight_decay=5e-4)

    def predict(self, image: Image) -> Image:
        image: torch.Tensor = resize(pil_to_tensor(image), size=(448, 448))
        
        with torch.inference_mode():
            self.eval()
            predictions = self(image.unsqueeze(0) / 255.)
            predictions = postprocess_outputs(transform_from_yolo(predictions))[0]

        return to_pil_image(draw_bounding_boxes(
            image=image,
            boxes=predictions[:, :4],
            labels=list(map(str, list(predictions[:, 5:])))
        ))

if __name__ == '__main__':
    yolo = FastYOLO1()
    im = torch.rand(5,3,448,448)
    out = yolo(im)
    print(sum(p.numel() for p in yolo.parameters()))
    print(out.shape)
    print(out[..., :10])
    # print(yolo)