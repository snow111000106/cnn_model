import torch.nn.functional as F
from config import train_loader
from model import Digit
import torch.optim as optim
import torch


def train_model(model, optimizer, epoch, train_loader):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to('cpu'), target.to('cpu')
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()
        if batch_index % 3000 == 0:
            print('train epoch:{}\t Loss:{:.6f}'.format(epoch, loss.item()))


if __name__ == '__main__':
    model = Digit().to("cpu")
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, 5):
        train_model(model, optimizer, epoch, train_loader)
    torch.save(model, 'model/mnist_model.h5')