import torch
x = torch.arange(9)
x_3x3= x.view(3,3)
x_3x3 = x.reshape(3,3)
print(x_3x3)
y = x_3x3.t()
print(y.shape)
print(y.contiguous().view(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2),dim=0).shape)
print(torch.cat((x1,x2),dim=1).shape)

print(x2.view(-1).shape)

batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch,-1)

z= x.permute(0,2,1)

x = torch.arange(10)
print(x.unsqueeze(0))
print(x.unsqueeze(1))