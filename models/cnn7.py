'''LeNet in PyTorch.'''
import math
import torch.nn as nn


class CNN7(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN7, self).__init__()
        self.depth = 32
        self.conv1 = nn.Conv2d(3,
                               self.depth,
                               kernel_size=3,
                               padding=2,
                               dilation=1)
        self.conv2 = nn.Conv2d(self.depth,
                               self.depth,
                               kernel_size=3,
                               padding=1,
                               dilation=1)
        self.conv3 = nn.Conv2d(self.depth,
                               self.depth * 2,
                               kernel_size=3,
                               padding=1,
                               dilation=1)
        self.conv4 = nn.Conv2d(self.depth * 2,
                               self.depth * 2,
                               kernel_size=3,
                               padding=1,
                               dilation=1)
        self.fc1 = nn.Linear(
            self.depth * 2 * math.ceil((16000 + 2) / (3 * 3 * 3)), num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.depth)
        self.bn2 = nn.BatchNorm2d(self.depth * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if __debug__:
            print(out.shape)
        out = self.maxpool(out)
        if __debug__:
            print(out.shape)
        out = self.relu(self.bn1(self.conv2(out)))
        if __debug__:
            print(out.shape)
        out = self.relu(self.bn1(self.conv2(out)))
        if __debug__:
            print(out.shape)
        out = self.relu(self.bn1(self.conv2(out)))
        if __debug__:
            print(out.shape)
        out = self.relu(self.bn2(self.conv3(out)))
        if __debug__:
            print(out.shape)
        out = self.maxpool(out)
        if __debug__:
            print(out.shape)
        out = self.relu(self.bn2(self.conv4(out)))
        if __debug__:
            print(out.shape)
        out = self.relu(self.bn2(self.conv4(out)))
        if __debug__:
            print(out.shape)
        out = self.maxpool(out)
        if __debug__:
            print(out.shape)
        out = out.view(out.size(0), -1)
        if __debug__:
            print(out.shape)
        out = self.fc1(out)
        if __debug__:
            print(out.shape)
        return out
