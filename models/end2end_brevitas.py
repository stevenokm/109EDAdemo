from brevitas.nn import QuantLinear, QuantReLU, QuantMaxPool1d, QuantConv1d
from brevitas.quant.binary import SignedBinaryWeightPerTensorConst
from brevitas.quant.binary import SignedBinaryActPerTensorConst
import torch.nn as nn

__all__ = ['end2end_brevitas']


class end2end_brevitas(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3):
        super(end2end_brevitas, self).__init__()
        self.weight_quant = SignedBinaryWeightPerTensorConst
        self.act_quant = SignedBinaryActPerTensorConst
        self.bias_quant = SignedBinaryWeightPerTensorConst
        self.input_quant = SignedBinaryActPerTensorConst
        self.output_quant = SignedBinaryActPerTensorConst
        self.ratioInfl = 1
        self.convDepth1 = 16
        self.convDepth2 = 32
        self.convDepth3 = 64
        self.convDepth4 = 128
        self.fcDepth = 4096
        self.embedding_factor = int(102400 // 100)
        self.pullSize1 = 8
        self.pullSize2 = 8
        self.features = nn.Sequential(
            # nn.BatchNorm1d(input_channels),
            # QuantHardTanh(act_quant=self.act_quant),
            QuantConv1d(input_channels,
                        int(self.convDepth1 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=64,
                        stride=2),
            # nn.BatchNorm1d(int(self.convDepth1 * self.ratioInfl)),
            QuantReLU(act_quant=self.act_quant),
            QuantMaxPool1d(kernel_size=self.pullSize1, stride=self.pullSize1),
            QuantConv1d(int(self.convDepth1 * self.ratioInfl),
                        int(self.convDepth2 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=32,
                        stride=2),
            # nn.BatchNorm1d(int(self.convDepth2 * self.ratioInfl)),
            QuantReLU(act_quant=self.act_quant),
            QuantMaxPool1d(kernel_size=self.pullSize2),
            QuantConv1d(int(self.convDepth2 * self.ratioInfl),
                        int(self.convDepth3 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=16,
                        stride=2),
            # nn.BatchNorm1d(int(self.convDepth3 * self.ratioInfl)),
            QuantReLU(act_quant=self.act_quant),
            QuantConv1d(int(self.convDepth3 * self.ratioInfl),
                        int(self.convDepth4 * self.ratioInfl),
                        weight_quant=self.weight_quant,
                        kernel_size=8,
                        stride=2),
            # nn.BatchNorm1d(int(self.convDepth4 * self.ratioInfl)),
            QuantReLU(act_quant=self.act_quant),
        )
        self.classifier = nn.Sequential(
            QuantLinear(self.embedding_factor,
                        self.fcDepth,
                        bias=False,
                        weight_quant=self.weight_quant),
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(self.fcDepth),
            QuantReLU(act_quant=self.act_quant),
            QuantLinear(self.fcDepth,
                        num_classes,
                        bias=False,
                        weight_quant=self.weight_quant),
            # nn.BatchNorm1d(num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        if __debug__:
            print(x.shape)
        x = x.view(-1, self.embedding_factor)
        x = self.classifier(x)
        return x
