import torch
import torch.nn.functional as F
from config import test_loader
from model import Digit
import torch.optim as optim


def model_test(model, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cpu'), target.to('cpu')
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("test - average loss:{:.4f},accuracy:{:.3f}\n".format(
            test_loss, 100.0*correct/len(test_loader.dataset)
        ))


if __name__ == '__main__':
    model = torch.load('model/mnist_model.h5', weights_only=False)
    optimizer = optim.Adam(model.parameters())
    model_test(model, test_loader)