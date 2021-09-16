import torch.nn as nn

from brevitas.nn import QuantLinear, QuantHardTanh, QuantMaxPool2d, QuantConv2d
from brevitas.quant.binary import SignedBinaryActPerTensorConst
from brevitas.quant.binary import SignedBinaryWeightPerTensorConst

from wsconv import WSConv2d

NegBiasLayer = nn.Identity

__all__ = ['M5_brevitas']


class M5_BN_brevitas(nn.Module):
    def __init__(self,
                 input_channels=1,
                 num_classes=35,
                 stride=4,
                 n_channel=128):
        super().__init__()
        self.emb_factor = (249, 1)
        self.n_channel = n_channel
        self.weight_quant = SignedBinaryWeightPerTensorConst
        self.act_quant = SignedBinaryActPerTensorConst
        self.conv1 = QuantConv2d(input_channels,
                                 self.n_channel,
                                 kernel_size=(84, 1),
                                 stride=stride,
                                 weight_quant=self.weight_quant)
        self.bn1 = nn.BatchNorm2d(self.n_channel)
        self.pool1 = QuantMaxPool2d((2, 1))
        self.conv2 = QuantConv2d(self.n_channel,
                                 self.n_channel,
                                 dilation=(1, 1),
                                 padding=(2, 0),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        self.bn2 = nn.BatchNorm2d(self.n_channel)
        self.pool2 = QuantMaxPool2d((2, 1))
        self.conv3 = QuantConv2d(self.n_channel,
                                 2 * self.n_channel,
                                 padding=(1, 0),
                                 dilation=(1, 1),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        self.bn3 = nn.BatchNorm2d(2 * self.n_channel)
        self.pool3 = QuantMaxPool2d((2, 1))
        # self.conv4 = QuantConv2d(2 * self.n_channel,
        #                          2 * self.n_channel,
        #                          padding=(1, 0),
        #                          kernel_size=(3, 1),
        #                          weight_quant=self.weight_quant)
        # self.bn4 = nn.BatchNorm2d(2 * self.n_channel)
        # self.pool4 = QuantMaxPool2d((4, 1))
        self.conv5 = QuantConv2d(2 * self.n_channel,
                                 4 * self.n_channel,
                                 padding=(2, 0),
                                 dilation=(1, 1),
                                 kernel_size=(3, 1),
                                 weight_quant=self.weight_quant)
        self.bn5 = nn.BatchNorm2d(4 * self.n_channel)
        self.pool5 = QuantMaxPool2d((2, 1))
        # self.conv6 = QuantConv2d(4 * self.n_channel,
        #                          4 * self.n_channel,
        #                          padding=(1, 0),
        #                          kernel_size=(3, 1),
        #                          weight_quant=self.weight_quant)
        # self.bn6 = nn.BatchNorm2d(4 * self.n_channel)
        # self.pool6 = QuantMaxPool2d((4, 1))
        # self.fc1 = QuantLinear(4 * self.n_channel * self.emb_factor[0],
        #                        4 * self.n_channel,
        #                        bias=False,
        #                        weight_quant=self.weight_quant)
        # self.bnfc1 = nn.BatchNorm1d(4 * self.n_channel)
        self.fc2 = QuantLinear(4 * self.n_channel * self.emb_factor[0],
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
        self.actfc2 = QuantHardTanh(act_quant=self.act_quant)
        # self.emb = QuantConv2d(4 * self.n_channel,
        #                        4 * self.n_channel,
        #                        kernel_size=self.emb_factor,
        #                        weight_quant=self.weight_quant)
        self.emb = nn.Identity()

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
        x = x.view(-1, 4 * self.n_channel * self.emb_factor[0])
        if __debug__:
            print(x.shape)
        # x = self.fc1(x)
        # x = self.actfc1(self.bnfc1(x))
        x = self.fc2(x)
        # x = self.actfc2(x)
        return x


class M5_NOBN_brevitas(nn.Module):
    def __init__(self,
                 input_channels=1,
                 num_classes=35,
                 stride=4,
                 n_channel=128):
        super().__init__()
        self.emb_factor = (249, 1)
        self.n_channel = n_channel
        self.weight_quant = SignedBinaryWeightPerTensorConst
        self.act_quant = SignedBinaryActPerTensorConst
        self.conv1 = WSConv2d(input_channels,
                                 self.n_channel,
                                 kernel_size=(84, 1),
                                 stride=stride,
                                 weight_quant=self.weight_quant)
        self.bn1 = NegBiasLayer(self.n_channel)
        self.pool1 = QuantMaxPool2d((2, 1))
        self.conv2 = WSConv2d(self.n_channel,
                                 self.n_channel,
                                 dilation=(1, 1),
                                 padding=(2, 0),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        self.bn2 = NegBiasLayer(self.n_channel)
        self.pool2 = QuantMaxPool2d((2, 1))
        self.conv3 = WSConv2d(self.n_channel,
                                 2 * self.n_channel,
                                 padding=(1, 0),
                                 dilation=(1, 1),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        self.bn3 = NegBiasLayer(2 * self.n_channel)
        self.pool3 = QuantMaxPool2d((2, 1))
        # self.conv4 = WSConv2d(2 * self.n_channel,
        #                          2 * self.n_channel,
        #                          padding=(1, 0),
        #                          kernel_size=(3, 1),
        #                          weight_quant=self.weight_quant)
        # self.bn4 = NegBiasLayer(2 * self.n_channel)
        # self.pool4 = QuantMaxPool2d((4, 1))
        self.conv5 = WSConv2d(2 * self.n_channel,
                                 4 * self.n_channel,
                                 padding=(2, 0),
                                 dilation=(1, 1),
                                 kernel_size=(3, 1),
                                 weight_quant=self.weight_quant)
        self.bn5 = NegBiasLayer(4 * self.n_channel)
        self.pool5 = QuantMaxPool2d((2, 1))
        # self.conv6 = WSConv2d(4 * self.n_channel,
        #                          4 * self.n_channel,
        #                          padding=(1, 0),
        #                          kernel_size=(3, 1),
        #                          weight_quant=self.weight_quant)
        # self.bn6 = NegBiasLayer(4 * self.n_channel)
        # self.pool6 = QuantMaxPool2d((4, 1))
        # self.fc1 = QuantLinear(4 * self.n_channel * self.emb_factor[0],
        #                        4 * self.n_channel,
        #                        bias=False,
        #                        weight_quant=self.weight_quant)
        # self.bnfc1 = nn.BatchNorm1d(4 * self.n_channel)
        self.fc2 = QuantLinear(4 * self.n_channel * self.emb_factor[0],
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
        self.actfc2 = QuantHardTanh(act_quant=self.act_quant)
        # self.emb = WSConv2d(4 * self.n_channel,
        #                        4 * self.n_channel,
        #                        kernel_size=self.emb_factor,
        #                        weight_quant=self.weight_quant)
        self.emb = nn.Identity()

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
        x = x.view(-1, 4 * self.n_channel * self.emb_factor[0])
        if __debug__:
            print(x.shape)
        # x = self.fc1(x)
        # x = self.actfc1(self.bnfc1(x))
        x = self.fc2(x)
        # x = self.actfc2(x)
        return x


def M5_brevitas(**kwargs):
    num_classes = kwargs.get('num_classes', 1000)
    input_channels = kwargs.get('input_channels', 3)
    n_channel = kwargs.get('n_channel', 128)
    stride = kwargs.get('stride', 4)
    batchnorm = kwargs.get('batchnorm', True)
    if batchnorm:
        return M5_BN_brevitas(num_classes=num_classes,
                              input_channels=input_channels,
                              n_channel=n_channel,
                              stride=stride)
    else:
        return M5_NOBN_brevitas(num_classes=num_classes,
                                input_channels=input_channels,
                                n_channel=n_channel,
                                stride=stride)
