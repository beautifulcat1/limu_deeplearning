import torch

X = torch.tensor([1.0,2.0],requires_grad=True)

Y = X + 1
Z = 2*Y**2
J = Z.sum()
J.backward()

print(Z)