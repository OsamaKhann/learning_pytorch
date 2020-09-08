#### Tensor Indexing ####
import torch

batch_size = 10
features = 25

x =  torch.randn((batch_size, features))
print(x.shape)
print(x[0].shape)
print(x[:,1].shape)
print(x[:3,1:4])

# Fancy Indexing
x = torch.arange(10)
indices = [2, 4, 6, 9]
print(x[indices])

x = torch.randn((3, 3))
rows = torch.tensor([0, 1])
cols = torch.tensor([0, 2])
print(x[(rows,cols)])

# More Advance indexing
x = torch.arange(10)
print(x[(x > 2) & (x < 8)])
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])

# Useful Operations
print(torch.where(x > 5,x, x*2))
print(torch.tensor([1,2,1,1,1,1,2,2,3,3,4,2,4,6,5,5]).unique())
print(x.ndimension())
print(x.numel())