import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import BinarizeLinear, BinarizeConv1d

__all__ = ['alexnet_binary']


class AlexNetOWT_BN(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3):
        super(AlexNetOWT_BN, self).__init__()
        self.ratioInfl = 1
        self.convDepth = 32
        self.fcDepth = 16
        self.embedding_factor = 87360 // 32
        self.cell_kernel_size = 41
        self.features = nn.Sequential(
            #nn.BatchNorm1d(input_channels),
            nn.Hardtanh(inplace=True),
            BinarizeConv1d(input_channels,
                           int(self.convDepth * self.ratioInfl),
                           kernel_size=self.cell_kernel_size,
                           dilation=1),
            #nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(int(self.convDepth * self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            BinarizeConv1d(int(self.convDepth * self.ratioInfl),
                           int(self.convDepth * self.ratioInfl),
                           kernel_size=self.cell_kernel_size,
                           dilation=2),
            #nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(int(self.convDepth * self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            BinarizeConv1d(int(self.convDepth * self.ratioInfl),
                           int(self.convDepth * self.ratioInfl),
                           kernel_size=self.cell_kernel_size,
                           dilation=4),
            #nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(int(self.convDepth * self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            BinarizeConv1d(int(self.convDepth * self.ratioInfl),
                           int(self.convDepth * self.ratioInfl),
                           kernel_size=self.cell_kernel_size,
                           dilation=8),
            #nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(int(self.convDepth * self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            BinarizeConv1d(int(self.convDepth * self.ratioInfl),
                           int(self.convDepth * self.ratioInfl),
                           kernel_size=self.cell_kernel_size,
                           dilation=16),
            #nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(int(self.convDepth * self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            BinarizeConv1d(int(self.convDepth * self.ratioInfl),
                           int(self.convDepth * self.ratioInfl),
                           kernel_size=self.cell_kernel_size,
                           dilation=32),
            #nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(int(self.convDepth * self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            BinarizeConv1d(int(self.convDepth * self.ratioInfl),
                           int(self.fcDepth * self.ratioInfl),
                           kernel_size=self.cell_kernel_size,
                           dilation=64),
            nn.MaxPool1d(kernel_size=4))
        self.classifier = nn.Sequential(
            #BinarizeLinear(self.fcDepth * self.embedding_factor,
            #               self.fcDepth * 128),
            #nn.BatchNorm1d(self.fcDepth * 128),
            #nn.Hardtanh(inplace=True),
            #BinarizeLinear(self.fcDepth * 128,
            #               self.fcDepth * 32),
            #nn.BatchNorm1d(self.fcDepth * 32),
            #nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            #BinarizeLinear(self.fcDepth * 32, num_classes),
            BinarizeLinear(self.fcDepth * self.embedding_factor, num_classes),
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
        self.regime = {
            0: {
                'optimizer': 'Adam',
                'lr': 5e-3
            },
            20: {
                'lr': 1e-3
            },
            30: {
                'lr': 5e-4
            },
            35: {
                'lr': 1e-4
            },
            40: {
                'lr': 1e-5
            }
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train':
            transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize
            ]),
            'eval':
            transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(), normalize
            ])
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.fcDepth * self.embedding_factor)
        x = self.classifier(x)
        return x


def alexnet_binary(**kwargs):
    num_classes = kwargs.get('num_classes', 1000)
    return AlexNetOWT_BN(num_classes)
