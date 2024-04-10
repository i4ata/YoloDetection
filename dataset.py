import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection

from lightning import LightningDataModule

import albumentations as A
import numpy as np

from typing import Tuple
import os

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
        self.image_size = 448
        self.transform = A.Compose(
            [A.Resize(self.image_size, self.image_size)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
        self.classes = ('person', 
                        'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
                        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor')

    def __len__(self) -> int:
        return len(self.pascal_voc)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        image, annotations = self.pascal_voc[index]
        
        boxes, labels = zip(*[
            (list(map(int, object['bndbox'].values())), object['name']) 
            for object in annotations['annotation']['object']
        ])
        
        transform = self.transform(image=np.asarray(image), bboxes=boxes, labels=labels)

        # Setting up the image
        image = torch.from_numpy(transform['image']).permute(2,0,1) / 255.

        # Setting up the targets
        # Transform boxes from [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height]
        boxes = torch.tensor(transform['bboxes']).float()
        x, y = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        x, y = x / 64, y / 64

        w, h =  boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        w, h = w / self.image_size, h / self.image_size

        objectness_score = torch.ones(len(boxes))
        labels = torch.tensor(list(map(self.classes.index, transform['labels'])))

        feature_map = torch.zeros(7, 7, 6)
        feature_map[torch.floor(x).long(), torch.floor(y).long()] = torch.stack(
            (torch.frac(x), torch.frac(y), w, h, objectness_score, labels), dim=1
        )

        return image, feature_map

if __name__ == '__main__':
    test_dataset = YOLODataset('data/train')
    d_loader = DataLoader(test_dataset, batch_size=2)
    x,y = next(iter(d_loader))
    print(x.shape, y.shape)
    print(y[0])