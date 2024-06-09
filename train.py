import torch
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm.auto import tqdm

from dataset import YOLODataset
from fast_yolov1 import FastYOLO1
from loss import YOLOv1Loss
from utils import transform_to_yolo

class Trainer:
    def __init__(self, model: FastYOLO1, name: str = 'default_name', device: str = 'cpu') -> None:
        self.model = model.to(device)
        self.name = name
        self.device = device

    def _train_step(self, train_dataloader: DataLoader) -> Tuple[float, float]:
        train_loss, train_map = 0, 0
        self.model.train()
        for images_batch, boxes_batch, labels_batch in train_dataloader:
            targets = torch.stack([
                transform_to_yolo(boxes=boxes.to(self.device), labels=labels.to(self.device)) 
                for boxes, labels in zip(boxes_batch, labels_batch)
            ])
            predictions = self.model(images_batch.to(self.device))
            loss = self.loss_fn(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss
        train_loss /= len(train_dataloader)
        return train_loss, train_map

    def _val_step(self, val_dataloader: DataLoader) -> Tuple[float, float]:
        val_loss, val_map = 0, 0
        self.model.eval()
        for images_batch, boxes_batch, labels_batch in val_dataloader:
            targets = torch.stack([
                transform_to_yolo(boxes=boxes.to(self.device), labels=labels.to(self.device)) 
                for boxes, labels in zip(boxes_batch, labels_batch)
            ])
            with torch.inference_mode():
                predictions = self.model(images_batch.to(self.device))
                loss = self.loss_fn(predictions, targets)
            val_loss += loss
        val_loss /= len(val_dataloader)
        return val_loss, val_map

    def fit(self, batch_size: int = 10, epochs: int = 10, learning_rate: float = .001) -> None:
        dataset = YOLODataset()
        train_dataloader, val_dataloader = dataset.get_dataloaders(batch_size=batch_size)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.loss_fn = YOLOv1Loss(n_boxes=self.model.n_boxes, n_classes=self.model.n_classes)

        for epoch in tqdm(range(epochs)):
            train_loss, train_map = self._train_step(train_dataloader)
            val_loss, val_map = self._val_step(val_dataloader)
            print(f'{epoch} | Train loss: {train_loss} | Val loss: {val_loss}', flush=True)

if __name__ == '__main__':
    model = FastYOLO1()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model=model, device=device)
    trainer.fit()
