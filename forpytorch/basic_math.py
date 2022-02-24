import torch

x = torch.tensor([-1,2,3])
y = torch.tensor([9,8,7])

# addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)
print(z1)

z2 = torch.add(x,y)
print(z2)

z3 = x+y
print(z3)

#subtraction
z = x-y
print(z)

#division
z = torch.true_divide(x,y)
print(z)

#inplace operations
t = torch.zeros(3)
t.add_(x) # all operations including "_" mean inplace operations
print(t)
t+=x
# exponentiation
z = x.pow(2)
z = x**2

#matrix multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

#matrix exponentation
matrix_exp = torch.rand(5,5)
print(matrix_exp.matrix_power(3))

#element wise mult
z = x*y
print(z)

# dot product
z = torch.dot(x,y)
print(z)

#batch matrix mult

batch = 3
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))

out_bmm = torch.bmm(tensor1,tensor2)# batch must be same,and the number of operation will equal to batch.

# example of broadcasting

x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1-x2
z = x1**x2
sum_x = torch.sum(x,dim=0)
values,indices = torch.max(x,dim=0)
values,indices = torch.min(x,dim=0)

abs_s = torch.abs(x)
print(abs_s)

z = torch.argmax(x,dim=0)# only one returned
z = torch.argmin(x,dim=0)
print(z)

mean_x = torch.mean(x.float(),dim=0)
print(mean_x)

z = torch.eq(x,y)
print(z)

sorted_y,indices = torch.sort(y,dim=0,descending=False)
print(sorted_y,indices)

a = torch.arange(9).reshape(3, 3)   # 创建3*3的tensor
b = torch.clamp(a, 3, 6)     # 对a的值进行限幅，限制在[3, 6]
print('a:', a)
print('shape of a:', a.shape)
print('b:', b)
print('shape of b:', b.shape)

x = torch.tensor([1,0,1,0,1],dtype=torch.bool)
z = torch.any(x) #true
z = torch.all(x) #false





