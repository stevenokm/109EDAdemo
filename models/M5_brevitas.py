from brevitas.nn import QuantLinear, QuantHardTanh, QuantMaxPool1d, QuantConv1d
from brevitas.quant.binary import SignedBinaryActPerTensorConst
from brevitas.quant.binary import SignedBinaryWeightPerTensorConst
import torch.nn as nn

__all__ = ['M5_brevitas']


class M5_brevitas(nn.Module):
    def __init__(self,
                 input_channels=1,
                 num_classes=35,
                 stride=4,
                 n_channel=128):
        super().__init__()
        self.emb_factor = 14
        self.weight_quant = SignedBinaryWeightPerTensorConst
        self.act_quant = SignedBinaryActPerTensorConst
        self.conv1 = QuantConv1d(input_channels,
                                 n_channel,
                                 kernel_size=80,
                                 stride=stride,
                                 weight_quant=self.weight_quant)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = QuantMaxPool1d(4)
        self.conv2 = QuantConv1d(n_channel,
                                 n_channel,
                                 kernel_size=3,
                                 weight_quant=self.weight_quant)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = QuantMaxPool1d(4)
        self.conv3 = QuantConv1d(n_channel,
                                 2 * n_channel,
                                 kernel_size=3,
                                 weight_quant=self.weight_quant)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = QuantMaxPool1d(4)
        # self.conv4 = QuantConv1d(2 * n_channel,
        #                          2 * n_channel,
        #                          kernel_size=3,
        #                          weight_quant=self.weight_quant)
        # self.bn4 = nn.BatchNorm1d(2 * n_channel)
        # self.pool4 = QuantMaxPool1d(4)
        self.conv5 = QuantConv1d(2 * n_channel,
                                 4 * n_channel,
                                 kernel_size=3,
                                 weight_quant=self.weight_quant)
        self.bn5 = nn.BatchNorm1d(4 * n_channel)
        self.pool5 = QuantMaxPool1d(4)
        # self.conv6 = QuantConv1d(4 * n_channel,
        #                          4 * n_channel,
        #                          kernel_size=3,
        #                          weight_quant=self.weight_quant)
        # self.bn6 = nn.BatchNorm1d(4 * n_channel)
        # self.pool6 = QuantMaxPool1d(4)
        self.fc1 = QuantLinear(4 * n_channel,
                               2 * n_channel,
                               bias=False,
                               weight_quant=self.weight_quant)
        self.bnfc1 = nn.BatchNorm1d(2 * n_channel)
        self.fc2 = QuantLinear(2 * n_channel,
                               num_classes,
                               bias=False,
                               weight_quant=self.weight_quant)
        # NOTE: activiation must different instance for
        # MultiThreshol-Add absorption
        self.act1 = QuantHardTanh(act_quant=self.act_quant)
        self.act2 = QuantHardTanh(act_quant=self.act_quant)
        self.act3 = QuantHardTanh(act_quant=self.act_quant)
        self.act4 = QuantHardTanh(act_quant=self.act_quant)
        self.act5 = QuantHardTanh(act_quant=self.act_quant)
        self.act6 = QuantHardTanh(act_quant=self.act_quant)
        self.actfc1 = QuantHardTanh(act_quant=self.act_quant)
        self.emb = QuantConv1d(4 * n_channel,
                               4 * n_channel,
                               kernel_size=self.emb_factor,
                               weight_quant=self.weight_quant)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(self.bn3(x))
        x = self.pool3(x)
        # x = self.conv4(x)
        # x = self.act4(self.bn4(x))
        # x = self.pool4(x)
        x = self.conv5(x)
        x = self.act5(self.bn5(x))
        x = self.pool5(x)
        # x = self.conv6(x)
        # x = self.act6(self.bn6(x))
        # x = self.pool6(x)
        if __debug__:
            print(x.shape)
        x = self.emb(x)
        x = x.squeeze(dim=2)
        if __debug__:
            print(x.shape)
        x = self.fc1(x)
        x = self.actfc1(self.bnfc1(x))
        x = self.fc2(x)
        return x
