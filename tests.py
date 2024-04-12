from fast_yolov1 import FastYOLO1
from dataset import YOLODataModule
import torch
m = FastYOLO1.load_from_checkpoint('lightning_logs/version_8138887/checkpoints/epoch=19-step=20.ckpt', map_location='cpu')
#print(m)
image, boxes, labels = next(iter(YOLODataModule(num_workers=3).train_dataloader()))
with torch.inference_mode():
    out: torch.Tensor = m(image)[0]
    # loss = m.training_step((image, boxes, labels), 0) Indeed it's 0.576
    f_map = m._transform_to_yolo(boxes[0], labels[0])

    grid = torch.stack(torch.meshgrid(torch.arange(7), torch.arange(7), indexing='ij'), dim=-1)
    good_box = out[3,3,5:9]
    print(good_box, f_map[3,3])
    good_box[..., [0,1]] = (good_box[..., [0,1]] + grid[3,3]) * 64
    good_box[..., [2,3]] *= 448
    print(good_box)
    print(boxes[0])