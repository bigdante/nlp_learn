import torch

# import torch.nn.functional as F

# prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


# design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # y_pred = F.sigmoid(self.linear(x))
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

# construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
# binary cross entropy
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

# PyTorch
import torch
from torch import nn
from torch import optim
# For plotting
import matplotlib.pyplot as plt
# os
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

x_data = torch.Tensor([1., 2., 3.]).reshape(-1, 1)
y_data = torch.Tensor([0, 0, 1]).reshape(-1, 1)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net = LogisticRegressionModel()
net.apply(init_weight)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
num_epochs = 1000
for epoch in range(num_epochs):
    _y = net(x_data)
    loss = criterion(_y, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(net.linear.weight.data)

test_data = torch.Tensor(list(range(11))).reshape(-1, 1)
y_pre = net(test_data)
plt.plot(test_data.tolist(), y_pre.tolist())
plt.xlabel('x')
plt.plot([-0, 11], [0.5, 0.5], c='r')
plt.ylabel('pre y')
plt.grid()
plt.show()


