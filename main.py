import os
import torch
import matplotlib.pyplot as plt
from PIL import Image  # 导入PIL库，用于图像处理
import torch.nn as nn  # PyTorch中的神经网络模块
import torch.nn.functional as F  # 包含一些常用的神经网络操作，如ReLU、softmax等
import torch.optim as optim  # PyTorch中的优化器模块
from torchvision import datasets, transforms  # 用于处理数据集和图像转换
from torch.utils.data import DataLoader  # 用于批量加载数据集
import tensorflow as tf

# 禁用 TensorFlow 的一些优化
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 从 tensorflow 加载 mnist 数据的代码。
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()  # 加载 MNIST 数据集，使用 TensorFlow 方式
# print(x_train.shape)  # 打印训练数据的形状，原代码有误，应该是 x_train.shape
# save_dir = './data/images/'  # 设置保存图像的文件夹路径
# if not os.path.exists(save_dir):  # 如果该目录不存在
#     os.makedirs(save_dir)  # 创建该目录
#
# plt.figure()  # 创建一个新的图形
# for i in range(1, 31):  # 循环显示前 30 张图像
#     img = x_train[i].reshape((28, 28))  # 重新塑造图像为 28x28 的矩阵
#     plt.subplot(6, 5, i)  # 创建一个 6x5 的子图并显示第 i 张图像
#     plt.imshow(img, cmap='gray')  # 显示图像，使用灰度色图
#     plt.savefig(save_dir+'image_{}.png'.format(i))  # 保存图像为 PNG 文件
#     plt.xlabel('image_{}'.format(i))  # 在图像下方标注编号
#     plt.xticks([])  # 隐藏 x 轴的刻度
#     plt.yticks([])  # 隐藏 y 轴的刻度
#
# plt.show()  # 显示图形窗口

# 定义图像预处理流程
pipeline = transforms.Compose(
    [
        transforms.ToTensor(),  # 将图像转换为 Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 对 MNIST 数据进行标准化，均值和标准差为 (0.1307,) 和 (0.3081,)
    ]
)

# 加载 MNIST 训练和测试数据集
train_set = datasets.MNIST("data", train=True, download=False, transform=pipeline)  # 加载训练集
test_set = datasets.MNIST("data", train=False, download=False, transform=pipeline)  # 加载测试集

# 使用 DataLoader 对数据集进行批量加载
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)  # 训练集，batch_size=16，随机打乱
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)  # 测试集，batch_size=16，不打乱顺序


# 定义神经网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义卷积层，conv1 将输入的 1 通道图像转换为 10 通道，卷积核大小为 5x5
        self.conv1 = nn.Conv2d(1, 10, 5)
        # 第二个卷积层，conv2 将 10 通道的输入图像转换为 20 通道，卷积核大小为 3x3
        self.conv2 = nn.Conv2d(10, 20, 3)
        # 第一个全连接层，将 20*10*10 的输入展平并连接到 500 个神经元
        self.fc1 = nn.Linear(20*10*10, 500)
        # 第二个全连接层，将 500 个神经元连接到 10 个输出神经元，表示 0-9 类别
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0)  # 获取输入数据的批量大小
        x = self.conv1(x)  # 经过第一个卷积层
        x = F.relu(x)  # 使用 ReLU 激活函数
        x = F.max_pool2d(x, 2, 2)  # 使用 2x2 的最大池化层

        x = self.conv2(x)  # 经过第二个卷积层
        x = F.relu(x)  # 使用 ReLU 激活函数

        x = x.view(input_size, -1)  # 将输出展平，展成 1D 向量

        x = self.fc1(x)  # 经过第一个全连接层
        x = F.relu(x)  # 使用 ReLU 激活函数

        x = self.fc2(x)  # 经过第二个全连接层

        output = F.log_softmax(x, dim=1)  # 使用 LogSoftmax 获取每个类别的对数概率
        return output  # 返回输出


# 定义训练过程
def train_model(model, devices, train_loader, optimizer, epoch):
    model.train()  # 设置模型为训练模式
    for batch_index, (data, target) in enumerate(train_loader):  # 遍历训练集中的每个批次
        data, target = data.to(devices), target.to(devices)  # 将数据和标签转移到指定设备（GPU/CPU）
        optimizer.zero_grad()  # 清除梯度
        output = model(data)  # 获取模型的输出
        loss = F.cross_entropy(output, target)  # 计算交叉熵损失

        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新参数
        if batch_index % 3000 == 0:  # 每 3000 个批次打印一次损失值
            print('train epoch:{}\t Loss:{:.6f}'.format(epoch, loss.item()))


# 定义测试过程
def model_test(model, devices, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0.0  # 初始化正确分类的计数器
    test_loss = 0.0  # 初始化总测试损失
    with torch.no_grad():  # 在测试时禁用梯度计算
        for data, target in test_loader:  # 遍历测试集中的每个批次
            data, target = data.to(devices), target.to(devices)  # 将数据和标签转移到指定设备
            output = model(data)  # 获取模型的输出
            test_loss += F.cross_entropy(output, target).item()  # 计算损失并累加
            pred = output.max(1, keepdim=True)[1]  # 获取预测的标签（输出概率最大值对应的标签）
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计预测正确的样本数
        test_loss /= len(test_loader.dataset)  # 计算平均测试损失
        # 打印测试集的损失和准确率
        print("test - average loss:{:.4f}, accuracy:{:.3f}%\n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))

model = Digit().to("cpu")  # 实例化模型并将其放到 CPU 上
optimizer = optim.Adam(model.parameters())  # 使用 Adam 优化器
for epoch in range(1, 5):  # 训练 1 个 epoch
    train_model(model, "cpu", train_loader, optimizer, epoch)  # 训练模型
    # model_test(model, "cpu", test_loader)  # 评估模型
torch.save(model, 'mnist_model.h5')  # 保存训练好的模型到文件

# 加载已保存的模型
# model = torch.load('model/mnist_model.h5', weights_only=False)  # 设置 weights_only=False
#
# # 测试第 10 张测试图像
# data, target = test_set[10]  # 获取第 10 张测试图像及其标签
# plt.imshow(data.reshape((28, 28)), cmap='gray')  # 显示该图像，使用灰度色图
# pred = model(data)  # 使用模型进行预测
# print(pred[0], "预期为{}".format(pred[0].argmax()))  # 打印模型输出，并显示预测的标签
# plt.title(target)  # 将真实标签显示在图像的标题上
# plt.show()
