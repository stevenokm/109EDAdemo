from brevitas.nn import QuantLinear, QuantHardTanh, QuantMaxPool1d, QuantConv1d, QuantScaleBias
from brevitas.quant.binary import *
import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['alexnet_binary']


class AlexNetOWT_BN(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3):
        super(AlexNetOWT_BN, self).__init__()
        self.weight_quant = SignedBinaryWeightPerTensorConst
        self.act_quant = SignedBinaryActPerTensorConst
        self.ratioInfl = 0.5
        self.depth = 8
        self.embedding_factor = 4
        self.features = nn.Sequential(
            QuantConv1d(input_channels,
                        int(512 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=23,
                        dilation=7),
            QuantMaxPool1d(kernel_size=5, padding=2),
            QuantScaleBias(int(512 * self.ratioInfl),
                           bias=True,
                           weight_quant=self.weight_quant),
            QuantHardTanh(act_quant=self.act_quant),
            QuantConv1d(int(512 * self.ratioInfl),
                        int(512 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=13,
                        dilation=3),
            QuantMaxPool1d(kernel_size=3, padding=1),
            QuantScaleBias(int(512 * self.ratioInfl),
                           bias=True,
                           weight_quant=self.weight_quant),
            QuantHardTanh(act_quant=self.act_quant),
            QuantConv1d(int(512 * self.ratioInfl),
                        int(512 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=7,
                        dilation=2),
            QuantMaxPool1d(kernel_size=3, padding=1),
            QuantScaleBias(int(512 * self.ratioInfl),
                           bias=True,
                           weight_quant=self.weight_quant),
            QuantHardTanh(act_quant=self.act_quant),
            QuantConv1d(int(512 * self.ratioInfl),
                        int(512 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=5,
                        dilation=1),
            QuantScaleBias(int(512 * self.ratioInfl),
                           bias=True,
                           weight_quant=self.weight_quant),
            QuantHardTanh(act_quant=self.act_quant),
            QuantConv1d(int(512 * self.ratioInfl),
                        int(512 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=5,
                        dilation=1),
            QuantMaxPool1d(kernel_size=3, padding=1),
            QuantScaleBias(int(512 * self.ratioInfl),
                           bias=True,
                           weight_quant=self.weight_quant),
            QuantHardTanh(act_quant=self.act_quant),
            QuantConv1d(int(512 * self.ratioInfl),
                        self.depth,
                        weight_quant=self.weight_quant,
                        kernel_size=3,
                        dilation=1),
            #QuantConv1d(int(192 * self.ratioInfl),
            #               self.depth,
            #               kernel_size=5,
            #               dilation=2),
            QuantMaxPool1d(kernel_size=3, padding=1),
            QuantScaleBias(self.depth,
                           bias=True,
                           weight_quant=self.weight_quant),
            QuantHardTanh(act_quant=self.act_quant))
        self.classifier = nn.Sequential(
            QuantLinear(self.depth * self.embedding_factor,
                        self.depth * 64,
                        bias=False,
                        weight_quant=self.weight_quant),
            QuantScaleBias(self.depth * 64,
                           bias=True,
                           weight_quant=self.weight_quant),
            QuantHardTanh(act_quant=self.act_quant),
            #nn.Dropout(0.5),
            QuantLinear(self.depth * 64,
                        num_classes,
                        bias=False,
                        weight_quant=self.weight_quant),
            QuantScaleBias(num_classes,
                           bias=True,
                           weight_quant=self.weight_quant),
            #nn.LogSoftmax()
        )

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
        x = x.view(-1, self.depth * self.embedding_factor)
        x = self.classifier(x)
        return x


def alexnet_binary(**kwargs):
    num_classes = kwargs.get('num_classes', 1000)
    return AlexNetOWT_BN(num_classes)
