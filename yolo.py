from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision
import torch
import torch.nn as nn
from typing import List, Literal, Tuple, Dict
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, resize, to_pil_image
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import nms, box_convert
from abc import ABC, abstractmethod

class FastYOLO(LightningModule, ABC):
    n_classes: int
    n_boxes: int
    loss_fn: nn.Module
    net: nn.Module
    image_size: int
    losses: Dict[Literal['train', 'val'], List] = {'train': [], 'val': []}
    maps: Dict[Literal['train', 'val'], MeanAveragePrecision] = {
        'train': MeanAveragePrecision(box_format='cxcywh'), 
        'val': MeanAveragePrecision(box_format='cxcywh')
    }
    
    @abstractmethod
    def _step(self, batch:  Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]], stage: Literal['train', 'val']) -> torch.Tensor:
        pass
    
    @abstractmethod
    def _transform_from_yolo(self, detections: torch.Tensor) -> torch.Tensor:
        pass

    def _postprocess_outputs(self, detections: torch.Tensor, confidence_threshold: float = .5, iou_threshold: float = .3) -> List[torch.Tensor]:
        detections = [x[x[:, 4] > confidence_threshold] for x in detections]
        detections = [
            x[nms(
                boxes=box_convert(boxes=x[:, :4], in_fmt='cxcywh', out_fmt='xyxy'), 
                scores=x[:, 4], 
                iou_threshold=iou_threshold
            )] 
            for x in detections
        ]
        return detections

    def _get_map(
            self, 
            detections: torch.Tensor, 
            target_boxes: List[torch.Tensor], 
            target_labels: List[torch.Tensor], 
            stage: Literal['train', 'val'] = 'train',
            confidence_threshold: float = .5,
            iou_threshold: float = .3
        ):

        detections = self._transform_from_yolo(detections=detections.detach())
        detections = self._postprocess_outputs(detections=detections, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold)

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
    
    def predict(self, image: Image, confidence_threshold: float = .5, iou_threshold: float = .3) -> Image:
        image: torch.Tensor = resize(pil_to_tensor(image), size=(self.image_size, self.image_size))
        
        with torch.inference_mode():
            self.eval()
            predictions = self(image.unsqueeze(0) / 255.)
            predictions = self._postprocess_outputs(
                detections=self._transform_from_yolo(predictions), 
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )[0]

        return to_pil_image(draw_bounding_boxes(
            image=image,
            boxes=predictions[:, :4],
            labels=list(map(str, list(predictions[:, 5:])))
        ))