from lightning import Trainer

from fast_yolov1 import DetectionModel
from dataset import create_data_module, YOLODataset
from utils import YOLOv1Loss

from torch.utils.data import DataLoader

if __name__ == '__main__':
    print('imports done')
    model = DetectionModel().to('cuda')
    print('model initialized')
    # data = create_data_module(batch_size=1, num_workers=3)
    # print('datamodule initialized')
    # trainer = Trainer(fast_dev_run=1, accelerator='gpu')
    # print('trainer intitialized')
    # trainer.fit(model=model, datamodule=data)
    d = YOLODataset()
    dl = DataLoader(dataset=d, batch_size=1)
    x,y = next(iter(dl))
    for i, (x, y) in enumerate(dl):
        if i == 10: break
        x,y = x.to('cuda'),y.to('cuda')
        out = model(x)
        print(out.shape)
        print(y.shape)
        l = YOLOv1Loss()
        loss = l(out, y)
        print(loss)
        print('*' * 50)