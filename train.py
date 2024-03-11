from fast_yolov1 import FastYOLO1
from utils import YOLOv1Loss
from dataset import YOLODataset

from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = YOLODataset('data/train')
    test_dataset = YOLODataset('data/test')
    model = FastYOLO1().to(device=device)
    loss_fn = YOLOv1Loss()
    optimizer = torch.optim.Adam(params=model.parameters())
    train_dataloader = DataLoader(train_dataset, batch_size=10, num_workers=3, pin_memory=device=='cuda')
    test_dataloader = DataLoader(test_dataset, batch_size=10, num_workers=3, pin_memory=device=='cuda')
    for i in range(10):
        model.train()
        train_loss, test_loss = 0, 0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y, y_pred)
            train_loss += loss.item()
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
        model.eval()
        for X, y in test_dataloader:
            with torch.inference_mode():
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_fn(y, y_pred)
            test_loss += loss.item()
        print(f'Train loss {train_loss/len(train_dataloader)}, Test loss {test_loss/len(test_dataloader)}')
    

