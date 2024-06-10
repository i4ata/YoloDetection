import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection
import albumentations as A
import numpy as np
from typing import Tuple, List

class YOLODataset:
    """
    ***********************************************************************************************************
    Each batch of the dataloader is a tuple of 3 elements:

    1. A torch tensor of stacked images [batch_size, 3, 448, 448]
    
    2. A list of length [batch_size]. 
    Each element is a tensor with the stacked bounding boxes for that image [n_boxes, [cx, cy, w, h]]
    
    3. A list of length [batch_size]
    Each element is a tensor with the stacked class indices for each bounding box in that image [n_boxes, 1]
    ***********************************************************************************************************
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_dataset = VOC('data/train')
        self.val_dataset = VOC('data/test')

    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        images, boxes, labels = zip(*batch)
        return torch.stack(images), boxes, labels
    
    def get_dataloaders(self, batch_size: int = 10) -> Tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=self._collate_fn
        )
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn
        )
        return train_dataloader, val_dataloader

class VOC(Dataset):
    def __init__(self, root: str = 'data/train') -> None:
        super().__init__()
        self.pascal_voc = VOCDetection(root, year='2007', image_set=root.split('/')[1])
        self.transform = A.Compose(
            [A.Resize(448, 448)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
        self.classes = (
            'person', 
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        )

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
            torch.tensor(transform['bboxes']).float(),
            torch.tensor(list(map(self.classes.index, transform['labels'])))
        )

if __name__ == '__main__':
    
    d = YOLODataset()
    dl,_=d.get_dataloaders()
    images, boxes, labels = next(iter(dl))
    print(images.shape)
    print([b.shape for b in boxes])
    print([l.shape for l in labels])    