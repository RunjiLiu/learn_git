import torch
import sys
import numpy as np

device = torch.device('cuda')
# device = torch.device('cpu')
sys.path.append('/home/yuki/Desktop')
from cs231n.data_utils import load_CIFAR10

cifar10_dir = '/home/yuki/Desktop/cs231n/assignment1/cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
nums = 40000


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train = X_train.reshape([50000, -1])
X_test = X_test.reshape([X_test.shape[0], -1])
nums2 = X_test.shape[0]
hiden_size1 = 600
hiden_size2 = 10

index = np.random.choice(range(X_train.shape[0]), nums)
train_x = torch.from_numpy(X_train[index]).cuda()
train_y = torch.from_numpy(y_train[index]).cuda()
# from_numpy 是从np.array转换为tensor, Tensor()是将list转为tensor
test_x = torch.from_numpy(X_test).cuda()
test_y = torch.from_numpy(y_test).cuda()

w1 = torch.randn(X_train.shape[1], hiden_size1, device=device, requires_grad=True).double() * 0.01
w2 = torch.randn(hiden_size1, hiden_size2, device=device, requires_grad=True).double() * 0.01

w1 = w1.cuda()
w2 = w2.cuda()

lr = 0.1
train_x = (train_x - torch.mean(train_x)) / torch.std(train_x)
test_x = (test_x - torch.mean(test_x)) / torch.std(test_x)
for i in range(100):
    res1 = train_x.mm(w1)
    out1 = res1.clamp(min=0)
    res2 = out1.mm(w2)
    out2 = torch.exp(res2)
    loss = -torch.log(out2[range(nums), train_y] / torch.sum(out2, dim=1)).sum() / nums
    print(i, loss)
    w1.retain_grad()
    w2.retain_grad()
    loss.backward(retain_graph=True)

    with torch.no_grad():
        w2.data = w2.data - lr * w2.grad.data
        w1.data = w1.data - lr * w1.grad.data
        w1.grad.zero_()
        w2.grad.zero_()

res1 = test_x.mm(w1)
out1 = res1.clamp(min=0)
res2 = out1.mm(w2)
pred = torch.argmax(res2, dim=1)
acc = (np.double(torch.sum(pred == test_y)) / nums2)
print(acc)
