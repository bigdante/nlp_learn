#basci tensor operations

import torch

print(torch.__version__)
# print(torch.rand(2,3))
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,device=device,requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

#other common initialization methods
print(torch.empty(size=(3,3)))
print(torch.zeros((3,3)))
print(torch.rand((3,3)))
print(torch.eye(5,5))
print(torch.arange(1,10,2))
print(torch.linspace(1,10,100))
print(torch.empty(size=(2,5)).normal_(mean=0,std=1))
print(torch.empty(size=(1,5)).uniform_(0,1))
print(torch.diag(torch.ones(3)))

#how to initialize and convert to other types(int,float,double)

tensor = torch.arange(4)
print(tensor)
print(tensor.bool()) #boolean true or false
print(tensor.short())#int 16
print(tensor.long())#int64
print(tensor.half())#float16
print(tensor.float())#float32
print(tensor.double())#float64

#array to tensor conversion and vice-versa

import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()