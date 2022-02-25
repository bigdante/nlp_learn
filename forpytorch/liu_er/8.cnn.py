# PyTorch
import torch
from torch.utils import data
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch import optim
# For plotting
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
test_data = datasets.MNIST(root='./data', download=True, train=False, transform=transform)
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


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.batch_size = x.shape[0]
        x = self.pooling(self.relu(self.conv1(x)))
        x = self.pooling(self.relu((self.conv2(x))))
        x = x.reshape(self.batch_size, -1)

        _y = self.fc(x)
        return _y


def test(net, test_iter):
    with torch.no_grad():
        total = 0
        correct = 0
        for x, y in test_iter:
            _y = net(x)
            _, pred = torch.max(_y, dim=1)
            correct += (pred == y).sum().item()
            total += _y.shape[0]
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


