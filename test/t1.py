import torch

X = torch.tensor([1.0,2.0])
Y = torch.tensor([1.0,2.0])
Z = torch.matmul(X,Y)
print(Z)