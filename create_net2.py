import torch
import sys
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda')
sys.path.append('/home/yuki/Desktop')
from cs231n.data_utils import load_CIFAR10

cifar10_dir = '/home/yuki/Desktop/cs231n/assignment1/cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
nums = 4000
nums1 = 40000
nums2 = X_test[0]

X_train = X_train.reshape([X_train.shape[0], -1])
X_test = X_test.reshape([X_test.shape[0], -1])
hiden_size1 = 1000
hiden_size2 = 10

index = np.random.choice(range(X_train.shape[0]), nums1)
train_x = torch.from_numpy(X_train[index]).float().cuda()
train_y = torch.from_numpy(y_train[index]).cuda()
# from_numpy 是从np.array转换为tensor, Tensor()是将list转为tensor
test_x = torch.from_numpy(X_test).float().cuda()
test_y = torch.from_numpy(y_test).cuda()

train_x = (train_x - torch.mean(train_x)) / torch.std(train_x)
test_x = (test_x - torch.mean(test_x)) / torch.std(test_x)

net = torch.nn.Sequential(torch.nn.Linear(train_x.shape[1], hiden_size1),
                          torch.nn.ReLU(), torch.nn.Linear(hiden_size1, hiden_size2)).cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

losses = []
for i in range(1000):
    y_pred = net(train_x)
    loss = torch.nn.functional.cross_entropy(y_pred, train_y)
    losses.append(loss)
    print(i, loss.data, np.double(torch.sum(torch.argmax(net(test_x), dim=1) == test_y)) / test_x.shape[0])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

plt.plot(losses)
plt.show()
