import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
BATCH_SIZE = 16  # 批次大小
EPOCH = 2  # 遍历次数

pipeline = transforms.Compose(
    [
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ]
)
train_set = datasets.MNIST("data", train=True, download=False, transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=False, transform=pipeline)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)