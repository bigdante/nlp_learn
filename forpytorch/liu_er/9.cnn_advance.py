# PyTorch
# PyTorch
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils import data
from torch import optim
from torch import nn
# For plotting
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST(download=False, root='./data', train=True, transform=transform)
test_data = datasets.MNIST(download=False, root='./data', train=False, transform=transform)
batch_size = 64
train_iter = data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True
)
test_iter = data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False
)


class Inception(nn.Module):
    # Inception (B, C, H, W) -> (B, 88, H, W)
    def __init__(self, input_channels):
        super(Inception, self).__init__()
        self.branch_pool = nn.Conv2d(input_channels, 24, kernel_size=(1, 1))

        self.branch1x1 = nn.Conv2d(input_channels, 16, kernel_size=(1, 1))

        self.branch5x5_1 = nn.Conv2d(input_channels, 16, kernel_size=(1, 1))
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=(5, 5), padding=2)

        self.branch3x3_1 = nn.Conv2d(input_channels, 16, kernel_size=(1, 1))
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=(3, 3), padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        pooling = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        x1 = self.branch_pool(pooling(x))
        x2 = self.branch1x1(x)
        x3 = self.branch5x5_1(x)
        x3 = self.branch5x5_2(x3)
        x4 = self.branch3x3_1(x)
        x4 = self.branch3x3_2(x4)
        x4 = self.branch3x3_3(x4)

        return torch.cat([x1, x2, x3, x4], dim=1)


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(88, 20, kernel_size=(5, 5))

        self.incep1 = Inception(input_channels=10)
        self.incep2 = Inception(input_channels=20)

        self.pooling = nn.MaxPool2d(2)
        self.linear = nn.Linear(1408, 10)

        self.activate = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.activate(self.pooling(self.conv1(x)))
        x = self.incep1(x)
        x = self.activate(self.pooling(self.conv2(x)))
        x = self.incep2(x)
        x = x.reshape(batch_size, -1)
        x = self.linear(x)

        return x


def test(net, test_iter):
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_iter:
            _y = net(x)
            _, pre = torch.max(_y, dim=1)
            correct += (y == pre).sum().item()
            total += y.shape[0]
        return correct / total


net = Module()
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 20
cr_set = []

for epoch in range(num_epochs):
    for x, y in train_iter:
        _y = net(x)
        loss = criterion(_y, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cr = test(net, test_iter)
    cr_set.append(cr)
    print("第{}轮,正确率是{}".format(epoch + 1, cr))

plt.figure()
plt.plot(list(range(num_epochs)), cr_set)
plt.xlabel('Epoch')
plt.ylabel('Correct Rate')
plt.show()

