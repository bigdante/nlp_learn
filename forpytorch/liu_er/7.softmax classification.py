# PyTorch
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from torch import optim
# For plotting
import matplotlib.pyplot as plt

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
test_data = datasets.MNIST(root='./data', download=True, train=False, transform=transform)
train_iter = data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True
)
test_iter = data.DataLoader(
    dataset=test_data,
    shuffle=False,
    batch_size=batch_size
)


class Module(nn.Module):
    def __init__(self, activate):
        super(Module, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 10)
        self.activate = activate

    def forward(self, x):
        x = x.reshape(-1, 784)
        y1 = self.activate(self.l1(x))
        y2 = self.activate(self.l2(y1))
        y3 = self.activate(self.l3(y2))
        y4 = self.activate(self.l4(y3))
        y5 = self.l5(y4)
        return y5


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.1)


def test(net, test_iter):
    total = 0
    correct = 0
    for x, y in test_iter:
        with torch.no_grad():
            _y = net(x)
            _, predicted = torch.max(_y.data, dim=1)
            total += y.shape[0]
            correct += (predicted == y).sum().item()
    return correct / total


net1 = Module(nn.ReLU())
net2 = Module(torch.sigmoid)
net1.apply(init_weight)
net2.apply(init_weight)
lr = 0.01
optimizer1 = optim.SGD(net1.parameters(), lr=lr)
optimizer2 = optim.SGD(net2.parameters(), lr=0.03)
criterion = nn.CrossEntropyLoss()
num_epoch = 25
correct_rate1 = []
correct_rate2 = []
for epoch in range(num_epoch):
    for index, data in enumerate(train_iter, 0):
        x, y = data
        _y1 = net1(x)
        _y2 = net2(x)
        loss1 = criterion(_y1, y)
        loss2 = criterion(_y2, y)

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

    rate1 = test(net1, test_iter)
    correct_rate1.append(rate1)
    rate2 = test(net2, test_iter)
    correct_rate2.append(rate2)
    print('第{}轮，正确率为{} 和{}'.format(epoch + 1, rate1, rate2))

plt.figure()
plt.plot(list(range(num_epoch)), correct_rate1, c='b', label='Relu')
plt.plot(list(range(num_epoch)), correct_rate2, c='r', label='Sigmoid')
plt.xlabel('Epoch')
plt.ylabel('Correct Rate')
plt.legend()
plt.grid()
plt.show()


# 第2种
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # -1其实就是自动获取mini_batch
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 最后一层不做激活，不进行非线性变换


model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 获得一个批次的数据和标签
        inputs, target = data
        optimizer.zero_grad()
        # 获得模型预测结果(64, 10)
        outputs = model(inputs)
        # 交叉熵代价函数outputs(64,10),target（64）
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

