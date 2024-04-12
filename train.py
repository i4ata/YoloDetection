from lightning import Trainer

from fast_yolov1 import FastYOLO1
from dataset import YOLODataModule
import torch

if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    print('imports done')
    model = FastYOLO1()
    print('model initialized')
    data = YOLODataModule(batch_size=10, num_workers=3)
    print('datamodule initialized')
    trainer = Trainer(overfit_batches=1,accelerator='gpu', max_epochs=20, deterministic=True, log_every_n_steps=1)
    print('trainer intitialized')
    trainer.fit(model=model, datamodule=data)