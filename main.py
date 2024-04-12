import os
#import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.imag.shape)
# save_dir = './data/images/'
# if os.path.exists(save_dir) is False:
#     os.makedirs(save_dir)
# plt.figure()
# for i in range(1, 31):
#     img = x_train[i].reshape((28, 28))
#     plt.subplot(6, 5, i)
#     plt.imshow(img, cmap='gray')
#     plt.savefig(save_dir+'image_{}.png'.format(i))
#     plt.xlabel('image_{}'.format(i))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
pipeline = transforms.Compose(
    [
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ]
)
train_set = datasets.MNIST("data", train=True, download=False, transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=False, transform=pipeline)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)


class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(input_size, -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def train_model(model, devices, train_loader, optimizer, epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(devices), target.to(devices)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()
        if batch_index % 3000 == 0:
            print('train epoch:{}\t Loss:{:.6f}'.format(epoch, loss.item()))


def model_test(model, devices, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(devices), target.to(devices)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("test - average loss:{:.4f},accuracy:{:.3f}\n".format(
            test_loss, 100.0*correct/len(test_loader.dataset)
        ))


# model = Digit().to("cpu")
# optimizer = optim.Adam(model.parameters())
# for epoch in range(1, 2):
#     train_model(model, "cpu", train_loader, optimizer, epoch)
#     #model_test(model, "cpu", test_loader)
# torch.save(model, 'mnist_model.h5')
model = torch.load('model/mnist_model.h5')
data, target = test_set[10]
plt.imshow(data.reshape((28, 28)), cmap='gray')
pred = model(data)
print(pred[0], "预期为{}".format(pred[0].argmax()))
plt.title(target)
plt.show()