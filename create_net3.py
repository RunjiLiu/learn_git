import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tt
import torchvision.datasets as td
import matplotlib.pyplot as plt

device = torch.device('cuda')
nums = 49000
cifar10_dir = 'data/cifar-10-batches-py'
transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
data_train = td.CIFAR10(cifar10_dir, train=True, download=False, transform=transform)
data_val = td.CIFAR10(cifar10_dir, train=True, download=False, transform=transform)
loader_train = torch.utils.data.DataLoader(data_train, batch_size=64,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(range(nums)))
loader_val = torch.utils.data.DataLoader(data_val, batch_size=64,
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(range(nums, 50000)))


class my_net(torch.nn.Module):
    def __init__(self, channel_in, channel_1, channel_2, channel_3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channel_in, channel_1, 5, padding=2)
        torch.nn.init.kaiming_normal(self.conv1.weight)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(channel_1, channel_2, 3, padding=1)
        torch.nn.init.kaiming_normal(self.conv2.weight).cuda()
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(channel_2, channel_3, 3, padding=1)
        torch.nn.init.kaiming_normal(self.conv2.weight).cuda()
        self.relu3 = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(channel_3 * 32 * 32, 256)
        torch.nn.init.kaiming_normal(self.fc1.weight).cuda()
        self.relu4 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 128)
        torch.nn.init.kaiming_normal(self.fc2.weight).cuda()
        self.relu5 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(128, 10)
        torch.nn.init.kaiming_normal(self.fc3.weight).cuda()

    def forward(self, X):
        out1 = self.conv1(X)
        out2 = self.relu1(out1)
        out3 = self.conv2(out2)
        out4 = self.relu2(out3)
        out5 = self.conv3(out4)
        out6 = self.relu3(out5)
        out6 = out6.reshape(out6.shape[0], -1)
        out7 = self.fc1(out6)
        out8 = self.relu4(out7)
        out9 = self.fc2(out8)
        out10 = self.relu5(out9)
        out = self.fc3(out10)
        return out


losses = []
model = my_net(3, 32, 64, 128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 10

iter_num = 0
model.to(device=device)
for i in range(epochs):
    all_num = 0
    right_num = 0
    for x, y in loader_train:
        x = x.to(device=device)
        y = y.to(device=device)
        scores = model(x)
        loss = torch.nn.functional.cross_entropy(scores, y)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = scores.max(1)
        right_num += np.double(torch.sum(pred == y).data)
        all_num += x.shape[0]
        iter_num += 1
    print(f"epochs {i} ,acc {right_num / all_num}")

plt.plot(range(iter_num), losses, label="loss")
plt.legend()
plt.show()

right_num = 0
all_num = 0
with torch.no_grad():
    for x, y in loader_val:
        x = x.to(device=device)
        y = y.to(device=device)
        scores = model(x)
        _, pred = scores.max(1)
        right_num += torch.sum(pred == y)
        all_num += x.shape[0]
print(f"finally accuracy {np.double(right_num) / all_num:.2f}")
