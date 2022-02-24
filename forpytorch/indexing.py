import torch
batchz_size = 10
feature = 25
x = torch.rand(batchz_size,feature)
print(x) # x[0:]
print(x[0].shape)
print(x[:,0])
print(x[2,0:10])
x[0,0]=100
print(x)

#facy indexing
x= torch.arange(19)
indices = [2,5,4]
print(x[indices])

x = torch.tensor(([1,2,3],
                  [6,7,8],
                  [4,5,6]))
rows = torch.tensor([1,0])
cols = torch.tensor([2,1])
print(x[rows,cols])
# more advanced indexing
x = torch.arange(10)
print(x[(x<2)|(x>8)])
print(x[x.remainder(2)==0])

print(torch.where(x>5,x,x**2))
print(torch.tensor([0,0,1,2,2,3,4]).unique())

print(x.ndimension())
print(x.numel())

