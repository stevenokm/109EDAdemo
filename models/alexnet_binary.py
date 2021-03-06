import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import BinarizeLinear, BinarizeConv1d

__all__ = ['alexnet_binary']


class AlexNetOWT_BN(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3):
        super(AlexNetOWT_BN, self).__init__()
        self.ratioInfl = 1
        self.convDepth1 = 64
        self.convDepth2 = 128
        self.convDepth3 = 128
        self.fcDepth = 2048
        self.embedding_factor = int(19968 // 2)
        self.cell_kernel_size = 41
        self.pullSize1 = 3
        self.pullSize2 = 3
        self.features = nn.Sequential(
            nn.BatchNorm1d(input_channels),
            #nn.Hardtanh(inplace=True),
            BinarizeConv1d(input_channels,
                           int(self.convDepth1 * self.ratioInfl),
                           kernel_size=41,
                           dilation=1),
            nn.BatchNorm1d(int(self.convDepth1 * self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            nn.MaxPool1d(kernel_size=self.pullSize1),

            BinarizeConv1d(int(self.convDepth1 * self.ratioInfl),
                           int(self.convDepth1 * self.ratioInfl),
                           kernel_size=self.cell_kernel_size,
                           dilation=2),
            nn.BatchNorm1d(int(self.convDepth1 * self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            nn.MaxPool1d(kernel_size=self.pullSize2),

            BinarizeConv1d(int(self.convDepth1 * self.ratioInfl),
                           int(self.convDepth2 * self.ratioInfl),
                           kernel_size=self.cell_kernel_size,
                           dilation=2),
            nn.BatchNorm1d(int(self.convDepth2 * self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            nn.MaxPool1d(kernel_size=self.pullSize2),

            BinarizeConv1d(int(self.convDepth2 * self.ratioInfl),
                           int(self.convDepth2 * self.ratioInfl),
                           kernel_size=self.cell_kernel_size,
                           dilation=2),
            nn.BatchNorm1d(int(self.convDepth2 * self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            nn.MaxPool1d(kernel_size=self.pullSize2),

            BinarizeConv1d(int(self.convDepth2 * self.ratioInfl),
                           int(self.convDepth3 * self.ratioInfl),
                           kernel_size=self.cell_kernel_size,
                           dilation=2),
            nn.BatchNorm1d(int(self.convDepth3 * self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            #nn.MaxPool1d(kernel_size=self.pullSize))
        )
        self.classifier = nn.Sequential(
            BinarizeLinear(self.embedding_factor, self.fcDepth),
            #nn.Dropout(0.5),
            nn.BatchNorm1d(self.fcDepth),
            nn.Hardtanh(inplace=True),
            BinarizeLinear(self.fcDepth, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.LogSoftmax())

        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-2,
        #        'weight_decay': 5e-4, 'momentum': 0.9},
        #    10: {'lr': 5e-3},
        #    15: {'lr': 1e-3, 'weight_decay': 0},
        #    20: {'lr': 5e-4},
        #    25: {'lr': 1e-4}
        #}
        #self.regime = {
        #    0: {
        #        'optimizer': 'Adam',
        #        'lr': 5e-3
        #    },
        #    20: {
        #        'lr': 1e-3
        #    },
        #    30: {
        #        'lr': 5e-4
        #    },
        #    35: {
        #        'lr': 1e-4
        #    },
        #    40: {
        #        'lr': 1e-5
        #    }
        #}
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        #self.input_transform = {
        #    'train':
        #    transforms.Compose([
        #        transforms.Scale(256),
        #        transforms.RandomCrop(224),
        #        transforms.RandomHorizontalFlip(),
        #        transforms.ToTensor(), normalize
        #    ]),
        #    'eval':
        #    transforms.Compose([
        #        transforms.Scale(256),
        #        transforms.CenterCrop(224),
        #        transforms.ToTensor(), normalize
        #    ])
        #}

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.embedding_factor)
        x = self.classifier(x)
        return x


def alexnet_binary(**kwargs):
    num_classes = kwargs.get('num_classes', 1000)
    return AlexNetOWT_BN(num_classes)
