
# 随机梯度下降性能好，但是不能并行，因为后一个w依赖前一个样本计算出的结果
# 随机梯度下降可以并行，不过性能没有随机的好，因此我们才使用batch size。
import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def forward(w, x):
    return x * w


def cost(x_cor, y_cor, w):
    y_hat = forward(w, x_cor)
    loss = (y_hat - y_cor) ** 2
    return loss.sum() / len(x_cor)


def gradient(x_cor, y_cor, w):
    grad = 2 * x_cor * (w * x_cor - y_cor)
    return grad.sum() / len(x_cor)


x_data = numpy.array([1.0, 2.0, 3.0])
y_data = numpy.array([2.0, 4.0, 6.0])
num_epochs = 100
lr = 0.01
w_train = numpy.array([1.0])
epoch_cor = []
loss_cor = []
for epoch in range(num_epochs):
    mse_loss = cost(x_data, y_data, w_train)
    loss_cor.append(mse_loss)
    w_train -= lr * gradient(x_data, y_data, w_train)
    epoch_cor.append(epoch + 1)

plt.figure()
plt.plot(epoch_cor, loss_cor, c='b')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()


# 随机梯度下降，只用一个样本，防止进入鞍点不迭代状态
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


# calculate loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# define the gradient function  sgd
def gradient(x, y):
    return 2 * x * (x * w - y)


epoch_list = []
loss_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad  # update weight by every grad of sample of training set
        print("\tgrad:", x, y, grad)
        l = loss(x, y)
    print("progress:", epoch, "w=", w, "loss=", l)
    epoch_list.append(epoch)
    loss_list.append(l)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()



