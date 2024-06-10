import torch
from fast_yolov1 import FastYOLO1
from dataset import YOLODataset
from utils import transform_from_yolo
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = YOLODataset().val_dataset
    image, boxes, labels = d[2]

    model = FastYOLO1().to(device)
    model.load_state_dict(torch.load('models/model.pt', map_location=device))
    model.eval()
    print(boxes, labels)

    with torch.inference_mode():
        predictions = transform_from_yolo(model(image.to(device).unsqueeze(0)), objectness_threshold=.1)[0]
    print(predictions)
    image = draw_bounding_boxes(
        image=(image * 255.).to(torch.uint8),
        boxes=predictions['boxes'].cpu(),
        labels=list(map(lambda x: str(x.item()), predictions['labels'].cpu())),
        width=2,
        colors='red'
    ).permute(1,2,0)

    plt.imshow(image)
    plt.axis('off')
    plt.savefig('image.png', dpi=400)