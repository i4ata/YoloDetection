from fast_yolov1 import FastYOLO1
from dataset import YOLODataModule
import torch
from utils import transform_to_yolo, transform_from_yolo, postprocess_outputs
m = FastYOLO1.load_from_checkpoint('lightning_logs/version_8139749/checkpoints/epoch=19-step=20.ckpt', map_location='cpu')
#print(m)
image, boxes, labels = next(iter(YOLODataModule(num_workers=3).train_dataloader()))
with torch.inference_mode():
    out: torch.Tensor = m(image)[0]
    # loss = m.training_step((image, boxes, labels), 0) Indeed it's 0.576
    # f_map = transform_to_yolo(boxes[0], labels[0])

    out = out[..., 5:]
    out = transform_from_yolo(out.unsqueeze(0))
    out = postprocess_outputs(out)
    print([o.round(decimals=2) for o in out])