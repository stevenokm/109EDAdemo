import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import BinarizeLinear, BinarizeConv1d

__all__ = ['alexnet_binary']


class AlexNetOWT_BN(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3):
        super(AlexNetOWT_BN, self).__init__()
        self.ratioInfl = 3
        self.depth = 72
        self.embedding_factor = 352
        self.features = nn.Sequential(
            BinarizeConv1d(input_channels,
                           int(64 * self.ratioInfl),
                           kernel_size=5,
                           dilation=3),
            nn.MaxPool1d(kernel_size=5, padding=2),
            nn.BatchNorm1d(int(64 * self.ratioInfl)),
            nn.ReLU(inplace=True),
            BinarizeConv1d(int(64 * self.ratioInfl),
                           int(192 * self.ratioInfl),
                           kernel_size=5,
                           dilation=3),
            nn.MaxPool1d(kernel_size=3, padding=1),
            nn.BatchNorm1d(int(192 * self.ratioInfl)),
            nn.ReLU(inplace=True),
            BinarizeConv1d(int(192 * self.ratioInfl),
                           int(384 * self.ratioInfl),
                           kernel_size=5,
                           dilation=1),
            nn.BatchNorm1d(int(384 * self.ratioInfl)),
            nn.ReLU(inplace=True),
            BinarizeConv1d(int(384 * self.ratioInfl),
                           int(256 * self.ratioInfl),
                           kernel_size=3,
                           dilation=1),
            nn.BatchNorm1d(int(256 * self.ratioInfl)),
            nn.ReLU(inplace=True),
            BinarizeConv1d(int(256 * self.ratioInfl),
                           self.depth,
                           kernel_size=3,
                           dilation=1),
            #BinarizeConv1d(int(192 * self.ratioInfl),
            #               self.depth,
            #               kernel_size=3,
            #               padding=1),
            nn.MaxPool1d(kernel_size=3, padding=1),
            nn.BatchNorm1d(self.depth),
            nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            BinarizeLinear(self.depth * self.embedding_factor,
                           self.depth * 64),
            nn.BatchNorm1d(self.depth * 64),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(self.depth * 64, num_classes),
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
        x = x.view(-1, self.depth * self.embedding_factor)
        x = self.classifier(x)
        return x


def alexnet_binary(**kwargs):
    num_classes = kwargs.get('num_classes', 1000)
    return AlexNetOWT_BN(num_classes)
