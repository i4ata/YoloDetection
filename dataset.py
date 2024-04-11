from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection

from lightning import LightningDataModule

import albumentations as A
import numpy as np

from typing import Tuple, List
import os

class YOLODataModule(LightningDataModule):
    def __init__(self, batch_size: int = 10, num_workers: int = os.cpu_count()) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = YOLODataset('data/train')
        self.test_dataset = YOLODataset('data/test')

    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        images, boxes, labels = zip(*batch)
        return torch.stack(images), boxes, labels
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

class YOLODataset(Dataset):
    def __init__(self, root: str = 'data/train') -> None:
        super().__init__()
        self.pascal_voc = VOCDetection(root, year='2007', image_set=root.split('/')[1])
        self.transform = A.Compose(
            [A.Resize(448, 448)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
        self.classes = ('person', 
                        'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
                        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor')

    def __len__(self) -> int:
        return len(self.pascal_voc)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        image, annotations = self.pascal_voc[index]
        boxes, labels = zip(*[
            (list(map(float, object['bndbox'].values())), object['name']) 
            for object in annotations['annotation']['object']
        ])
        transform = self.transform(image=np.asarray(image), bboxes=boxes, labels=labels)

        return (
            torch.from_numpy(transform['image']).permute(2,0,1) / 255.,
            torch.tensor(transform['bboxes']),
            torch.tensor(list(map(self.classes.index, transform['labels'])))
        )

if __name__ == '__main__':
    test_dataset = YOLODataModule(num_workers=3)
    d_loader = test_dataset.train_dataloader()
    x,y = next(iter(d_loader))
    print(x.shape, y.shape)
    #print(y[0])