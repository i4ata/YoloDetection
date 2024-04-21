from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

from fast_yolov1 import FastYOLO1
from dataset import YOLODataModule

if __name__ == '__main__':
    model = FastYOLO1()
    data = YOLODataModule(batch_size=10, num_workers=7)
    early_stopper = EarlyStopping('val_loss')
    trainer = Trainer(
        overfit_batches=1, 
        accelerator='auto', 
        devices='auto', 
        max_epochs=20, 
        deterministic=True, 
        # callbacks=early_stopper
    )
    trainer.fit(model=model, datamodule=data)