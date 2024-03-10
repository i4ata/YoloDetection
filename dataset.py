from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from typing import Tuple, List
import numpy as np
import torch
from tqdm import tqdm

class YOLODataset(Dataset):
    def __init__(self, root: str = 'train') -> None:
        super().__init__()
        self.pascal_voc = VOCDetection(root, year='2007', image_set=root)
        self.image_size = 448
        self.transform = A.Compose(
            [A.Resize(self.image_size, self.image_size)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
        self.classes = ['person', 
                        'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
                        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    def __len__(self) -> int:
        return len(self.pascal_voc)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        image, annotations = self.pascal_voc[index]
        boxes, labels = [], []
        
        for object in annotations['annotation']['object']:
            boxes.append(list(map(int, object['bndbox'].values())))
            labels.append(object['name'])
        
        transform = self.transform(image=np.asarray(image), bboxes=boxes, labels=labels)

        # Setting up the image
        image = torch.tensor(transform['image']) / 255.

        # Setting up the targets
        # Transform boxes from [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height]
        boxes = torch.tensor(transform['bboxes']).float()
        x, y = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        w, h =  boxes[:, 2] - boxes[:, 0],       boxes[:, 3] - boxes[:, 1]

        objectness_score = torch.ones(len(boxes))
        labels = torch.tensor(list(map(self.classes.index, transform['labels'])))

        feature_map_dim = self.image_size // 7
        feature_map = torch.zeros(feature_map_dim, feature_map_dim, 6)
        feature_map[(x // 7).long(), (y // 7).long()] = torch.stack(
            (x, y, w, h, objectness_score, labels), dim=1
        )

        return image, feature_map

if __name__ == '__main__':
    test_dataset = YOLODataset('test')
    for i in tqdm(range(len(test_dataset))):
        a = test_dataset[i]
