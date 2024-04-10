from lightning import Trainer

from fast_yolov1 import DetectionModel
from dataset import create_data_module, YOLODataset
from utils import YOLOv1Loss

from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    print('imports done')
    model = DetectionModel().to('cuda')
    print('model initialized')
    data = create_data_module(batch_size=10, num_workers=3)
    print('datamodule initialized')
    trainer = Trainer(overfit_batches=1, accelerator='gpu')
    print('trainer intitialized')
    trainer.fit(model=model, datamodule=data)