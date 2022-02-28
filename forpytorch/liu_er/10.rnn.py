# PyTorch
import torch
from torch.utils import data
from torch import nn
from torch import optim
import torch.nn.functional as F
# For plotting
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# RNN hello -> ohlol

idx2char = ['e', 'h', 'l', 'o']
x_idx = torch.tensor([idx2char.index(i) for i in 'hello'])
# x_data = F.one_hot(x_idx, 4).reshape(-1, 1, 4).float()  # RNNCell
x_data = F.one_hot(x_idx, 4).float().reshape(5, 1, 4)  # 希望你能发现些什么, 诸如shape, type
print(x_data)
# y_data = torch.tensor([idx2char.index(i) for i in 'ohlol']).reshape(-1, 1) ## RNNCell
y_data = torch.tensor([idx2char.index(i) for i in 'ohlol'])
print(y_data)


class Module(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size=1, num_layer=1):
        super(Module, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layer = 1
        # self.rnncell = nn.RNNCell(input_size, hidden_size)  # RNNCell
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layer)

    def forward(self, x):
        hidden = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
        y, test = self.rnn(x, hidden)
        return y.reshape(-1, self.hidden_size), test[-1]
        # return self.rnncell(x, hidden)  # RNNCell

    # def init_hidden(self):  # RNNCell
    #     return torch.zeros(self.batch_size, self.hidden_size)


net = Module(4, 4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)
num_epochs = 10

for epoch in range(num_epochs):
    # RNNCell
    # loss = torch.tensor([0.])
    # hidden = net.init_hidden()  # RNNCell
    # for x, y in zip(x_data, y_data):
    #     # _y, hidden = net(x, hidden)
    #     loss += criterion(hidden, y)
    #     _, idx = torch.max(hidden, dim=1)
    #     print(idx2char[idx], end='')
    y, test = net(x_data)
    loss = criterion(y, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _, idx = torch.max(y, dim=1)
    pre = ''.join(idx2char[i] for i in idx)
    print(test.reshape(-1, 4) == y)  # 希望你能发现些什么
    print('{}, 第{}轮, loss为{:.4f}'.format(pre, epoch + 1, loss.item()))



