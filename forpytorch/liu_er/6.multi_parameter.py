import numpy as np
import torch
import matplotlib.pyplot as plt

# prepare dataset
xy = np.loadtxt('./PyTorch深度学习实践/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])  # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
print("input data.shape", x_data.shape)
y_data = torch.from_numpy(xy[:, [-1]])  # [-1] 最后得到的是个矩阵


# print(x_data.shape)
# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))  # y hat
        x = self.sigmoid(self.linear4(x))  # y hat
        return x


model = Model()

# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# training cycle forward, backward, update
for epoch in range(1000000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100000 == 99999:
        y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
        acc = torch.eq(y_pred_label, y_data).sum().item() / y_data.size(0)
        print("loss = ", loss.item(), "acc = ", acc)

# 第二种处理方式，数据的处理更加规范
# PyTorch
import torch
from torch import nn
from torch import optim
from torch.utils import data
# NumPy
import numpy as np


class DiabetesDataset(data.Dataset):
    def __init__(self, path):
        self.cor = np.loadtxt(path, delimiter=',', dtype=np.float32)
        self.len = self.cor.shape[0]
        self.x_data = torch.from_numpy(self.cor[:, :-1])
        self.y_data = torch.from_numpy(self.cor[:, -1]).reshape(-1, 1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    def __init__(self, input_dim, out_dim=1):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.activate = nn.ReLU()

    def forward(self, x):
        _y1 = self.activate(self.linear1(x))
        _y2 = self.activate(self.linear2(_y1))
        _y3 = torch.sigmoid(self.linear3(_y2))
        return _y3


batch_size = 62
train_data = DiabetesDataset('./PyTorch深度学习实践/diabetes.csv')
train_iter = data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1
)

net = Model(8)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)
num_epochs = 100
num_iter = len(train_data) // batch_size + 1 \
    if len(train_iter) % batch_size else len(train_data) // batch_size

if __name__ == '__main__':
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, y in train_iter:
            _y = net(x)
            loss = criterion(_y, y)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('第{}轮，loss为{:.8f}'.format(epoch + 1, epoch_loss / num_iter))






