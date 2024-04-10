import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection

from lightning import LightningDataModule

import albumentations as A
import numpy as np

from typing import Tuple
import os

from utils import transform_to_yolo

def create_data_module(batch_size: int = 10, num_workers: int = os.cpu_count()) -> LightningDataModule:
    return LightningDataModule.from_datasets(
        train_dataset=YOLODataset(root='data/train'),
        val_dataset=YOLODataset(root='data/test'),
        batch_size=batch_size,
        num_workers=num_workers
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

        return transform_to_yolo(
            image=transform['image'],
            boxes=np.array(transform['bboxes']),
            labels=list(map(self.classes.index, transform['labels']))
        )

if __name__ == '__main__':
    test_dataset = YOLODataset('data/train')
    d_loader = DataLoader(test_dataset, batch_size=1)
    x,y = next(iter(d_loader))
    print(x.shape, y.shape)
    #print(y[0])