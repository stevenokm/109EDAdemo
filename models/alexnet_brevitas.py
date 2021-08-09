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
        self.bias_quant = SignedBinaryWeightPerTensorConst
        self.input_quant = SignedBinaryActPerTensorConst
        self.output_quant = SignedBinaryActPerTensorConst
        self.ratioInfl = 1
        self.convDepth1 = 64
        self.convDepth2 = 256
        self.convDepth3 = 128
        self.fcDepth = 4096
        self.embedding_factor = int(1908736 // 32)
        self.cell_kernel_size = 3
        self.pullSize1 = 4
        self.pullSize2 = 2
        self.features = nn.Sequential(
            nn.BatchNorm1d(input_channels),
            #QuantHardTanh(act_quant=self.act_quant),
            QuantConv1d(input_channels,
                        int(self.convDepth1 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=64,
                        dilation=1),
            QuantConv1d(int(self.convDepth1 * self.ratioInfl),
                        int(self.convDepth1 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=64,
                        dilation=2),
            nn.BatchNorm1d(int(self.convDepth1 * self.ratioInfl)),
            QuantHardTanh(act_quant=self.act_quant),
            QuantMaxPool1d(kernel_size=self.pullSize1, stride=self.pullSize1),

            QuantConv1d(int(self.convDepth1 * self.ratioInfl),
                        int(self.convDepth1 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=self.cell_kernel_size,
                        dilation=2),
            QuantConv1d(int(self.convDepth1 * self.ratioInfl),
                        int(self.convDepth1 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=self.cell_kernel_size,
                        dilation=5),
            nn.BatchNorm1d(int(self.convDepth1 * self.ratioInfl)),
            QuantHardTanh(act_quant=self.act_quant),
            QuantMaxPool1d(kernel_size=self.pullSize2, stride=self.pullSize2),

            QuantConv1d(int(self.convDepth1 * self.ratioInfl),
                        int(self.convDepth2 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=self.cell_kernel_size,
                        dilation=1),
            QuantConv1d(int(self.convDepth2 * self.ratioInfl),
                        int(self.convDepth2 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=self.cell_kernel_size,
                        dilation=2),
            nn.BatchNorm1d(int(self.convDepth2 * self.ratioInfl)),
            QuantHardTanh(act_quant=self.act_quant),
            QuantMaxPool1d(kernel_size=self.pullSize2),

            QuantConv1d(int(self.convDepth2 * self.ratioInfl),
                        int(self.convDepth2 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=self.cell_kernel_size,
                        dilation=1),
            QuantConv1d(int(self.convDepth2 * self.ratioInfl),
                        int(self.convDepth2 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=self.cell_kernel_size,
                        dilation=2),
            QuantConv1d(int(self.convDepth2 * self.ratioInfl),
                        int(self.convDepth2 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=self.cell_kernel_size,
                        dilation=5),
            nn.BatchNorm1d(int(self.convDepth2 * self.ratioInfl)),
            QuantHardTanh(act_quant=self.act_quant),
            QuantMaxPool1d(kernel_size=self.pullSize2),

            QuantConv1d(int(self.convDepth2 * self.ratioInfl),
                        int(self.convDepth3 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=self.cell_kernel_size,
                        dilation=1),
            QuantConv1d(int(self.convDepth3 * self.ratioInfl),
                        int(self.convDepth3 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=self.cell_kernel_size,
                        dilation=2),
            QuantConv1d(int(self.convDepth3 * self.ratioInfl),
                        int(self.convDepth3 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=self.cell_kernel_size,
                        dilation=5),
            nn.BatchNorm1d(int(self.convDepth3 * self.ratioInfl)),
            QuantHardTanh(act_quant=self.act_quant),
            #QuantMaxPool1d(kernel_size=self.pullSize2),
        )
        self.classifier = nn.Sequential(
            QuantLinear(self.embedding_factor,
                        self.fcDepth,
                        bias=False,
                        weight_quant=self.weight_quant),
            #nn.Dropout(0.5),
            nn.BatchNorm1d(self.fcDepth),
            QuantHardTanh(act_quant=self.act_quant),
            QuantLinear(self.fcDepth,
                        num_classes,
                        bias=False,
                        weight_quant=self.weight_quant),
            nn.BatchNorm1d(num_classes),
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
        x = x.view(-1, self.embedding_factor)
        x = self.classifier(x)
        return x


def alexnet_binary(**kwargs):
    num_classes = kwargs.get('num_classes', 1000)
    return AlexNetOWT_BN(num_classes)
