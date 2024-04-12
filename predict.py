from model import Digit
import train
import config
import os
import torch
import test
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


if os.path.exists('model/mnist_model.h5'):
    model = torch.load('model/mnist_model.h5')
else:
    model = Digit().to("cpu")
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, config.EPOCH):
        train.train_model(model, config.train_loader, optimizer, epoch)
        test.model_test(model, config.test_loader)
    torch.save(model, 'model/mnist_model.h5')

pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图像
    #transforms.Resize((28, 28)),  # 调整图像大小为28x28像素
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])
path = 'data/images/test.png'
image = Image.open(path)
data = pipeline(image)

# data, target = config.test_set[4]
pred = model(data)

with torch.no_grad():
    result = pred[0].argmax()
    print(pred[0], result)
    pred = model(data)
    print("预测结果：", pred[0].argmax().item())
    plt.title(result)
    plt.imshow(data.reshape((28, 28)), cmap='gray')
    plt.show()