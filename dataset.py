from torchvision.datasets import VOCDetection

print('dataset imported')
dataset = VOCDetection('val', image_set='val', download=True)
print(dataset)
print(dataset[0])