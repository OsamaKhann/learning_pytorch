import torch

#### Initializing Tensor ####

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)


# Other common initialization methods
x = torch.empty(size=(3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3, 3))
x = torch.eye(3, 3)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1,  end=1, steps=10)
x = torch.empty((3, 3)).normal_(mean=0, std=1)
x = torch.empty((3, 3)).uniform_(0, 1)
x = torch.diag(torch.ones(3))
print(x)


# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())


# How to convert numpy array into tensor and vice-versa
import numpy as np
arr = np.ones((3, 3))
converted_tensor = torch.from_numpy(arr)
converted_arr = converted_tensor.numpy()

#### Tensor Maths & Comparison Operations #### 

x = torch.tensor([1, 3 , 6])
y = torch.tensor([2, 4 , 8])

# Addition
z_a = x + y

# Subtraction
z_s = x - y

# Divide 
z_d = torch.true_divide(x,y)

# Element wise Multiplication
z_m = x * y

# Dot Product Multiplication
z_dm = torch.dot(x,y)

# Exponentional 
z_e = x.pow(2)
z_e == x ** 2

# inplace operations
t_z = torch.zeros(3)
t_z.add_(x)

t_z += x

# Simple comparison
z_c = x > 0
z_c = x < 0

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2x3
 
x3 = x1.mm(x2)

# Matrix Exponentiation
m1 = torch.rand((5,5)) # should same dim
m1_e = m1.matrix_power(3)

# Batch Matrix Multiplication
batch = 32 
n = 10
m = 20
p = 30

bm1 = torch.rand((batch, m, n))
bm2 = torch.rand((batch, n, p))

out_bmm  = torch.bmm(bm1, bm2) # outputs: (batch, m, p)


# Example of Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2

z = x1 ** x2

# Other useful tensror operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)
z = torch.clamp(x, min=0)
x = torch.tensor([1, 0, 1, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)
z = torch.all(x)
