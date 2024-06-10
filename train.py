import torch
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm.auto import tqdm
from torchmetrics.detection import MeanAveragePrecision as MAP

from dataset import YOLODataset
from fast_yolov1 import FastYOLO1
from loss import YOLOv1Loss
from utils import transform_to_yolo, transform_from_yolo
from early_stopper import EarlyStopper

class Trainer:
    def __init__(self, model: FastYOLO1, name: str = 'default_name', device: str = 'cpu') -> None:
        self.model = model.to(device)
        self.name = name
        self.device = device

    def _train_step(self, train_dataloader: DataLoader) -> Tuple[float, float]:
        train_loss = 0
        self.model.train()
        self.train_map.reset()
        for images_batch, boxes_batch, labels_batch in train_dataloader:
            
            images_batch = images_batch.to(self.device)
            boxes_batch = [boxes.to(self.device) for boxes in boxes_batch]
            labels_batch = [labels.to(self.device) for labels in labels_batch]
            
            targets = torch.stack([
                transform_to_yolo(boxes=boxes, labels=labels) 
                for boxes, labels in zip(boxes_batch, labels_batch)
            ])
            
            predictions = self.model(images_batch)
            loss = self.loss_fn(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss

            self.train_map.update(
                preds=transform_from_yolo(predictions.detach(), objectness_threshold=.8, iou_threshold=.5),
                target=[{'boxes': boxes, 'labels': labels} for boxes, labels in zip(boxes_batch, labels_batch)]
            )

        train_loss /= len(train_dataloader)
        return train_loss, self.train_map.compute()['map'].item()

    def _val_step(self, val_dataloader: DataLoader) -> Tuple[float, float]:
        val_loss = 0
        self.model.eval()
        self.val_map.reset()
        for images_batch, boxes_batch, labels_batch in val_dataloader:

            images_batch = images_batch.to(self.device)
            boxes_batch = [boxes.to(self.device) for boxes in boxes_batch]
            labels_batch = [labels.to(self.device) for labels in labels_batch]

            targets = torch.stack([
                transform_to_yolo(boxes=boxes, labels=labels) 
                for boxes, labels in zip(boxes_batch, labels_batch)
            ])
            with torch.inference_mode():
                predictions = self.model(images_batch)
                loss = self.loss_fn(predictions, targets)
            val_loss += loss

            self.val_map.update(
                preds=transform_from_yolo(predictions.detach(), objectness_threshold=.8, iou_threshold=.5),
                target=[{'boxes': boxes, 'labels': labels} for boxes, labels in zip(boxes_batch, labels_batch)]
            )

        val_loss /= len(val_dataloader)
        return val_loss, self.val_map.compute()['map'].item()

    def fit(self, batch_size: int = 10, epochs: int = 10, learning_rate: float = .001) -> None:
        dataset = YOLODataset()
        train_dataloader, val_dataloader = dataset.get_dataloaders(batch_size=batch_size)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.loss_fn = YOLOv1Loss(n_boxes=self.model.n_boxes, n_classes=self.model.n_classes)
        self.train_map, self.val_map = MAP(), MAP()
        early_stopper = EarlyStopper()

        for epoch in tqdm(range(epochs)):
            train_loss, train_map = self._train_step(train_dataloader)
            val_loss, val_map = self._val_step(val_dataloader)
            print(f'{epoch} | Train loss: {train_loss} | Train map: {train_map} | Val loss: {val_loss} | Val map: {val_map}', flush=True)
            if early_stopper.check(val_loss):
                break
            if early_stopper.save_model: torch.save(self.model.state_dict(), 'models/' + self.name + '.pt')


if __name__ == '__main__':
    model = FastYOLO1()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model=model, name='model', device=device)
    trainer.fit(batch_size=8, learning_rate=.0005)
