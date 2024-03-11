import torch
a = torch.zeros(3,3,5)
print(a)
a[:,:,[0,1]] += torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), indexing='xy'), dim=2)
print(a)