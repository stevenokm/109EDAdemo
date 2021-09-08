from brevitas.nn import QuantLinear, QuantHardTanh, QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.quant.binary import SignedBinaryActPerTensorConst
from brevitas.quant.binary import SignedBinaryWeightPerTensorConst
import torch.nn as nn

__all__ = ['M5_brevitas']


class M5_BN_brevitas(nn.Module):
    def __init__(self,
                 input_channels=1,
                 num_classes=35,
                 stride=4,
                 n_channel=128):
        super().__init__()
        self.emb_factor = (16, 1)
        self.n_channel = n_channel
        self.weight_quant = SignedBinaryWeightPerTensorConst
        self.act_quant = SignedBinaryActPerTensorConst
        self.conv1 = QuantConv2d(input_channels,
                                 self.n_channel,
                                 bias=False,
                                 kernel_size=(84, 1),
                                 stride=stride,
                                 weight_quant=self.weight_quant)
        self.bn1 = nn.BatchNorm2d(self.n_channel)
        self.pool1 = nn.Sequential(
            QuantConv2d(self.n_channel,
                        self.n_channel,
                        bias=False,
                        stride=(4, 1),
                        kernel_size=(4, 1),
                        weight_quant=self.weight_quant),
            QuantHardTanh(act_quant=self.act_quant),
        )
        self.conv2 = QuantConv2d(self.n_channel,
                                 self.n_channel,
                                 bias=False,
                                 padding=(2, 0),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        self.bn2 = nn.BatchNorm2d(self.n_channel)
        self.pool2 = nn.Sequential(
            QuantConv2d(self.n_channel,
                        self.n_channel,
                        bias=False,
                        stride=(4, 1),
                        kernel_size=(4, 1),
                        weight_quant=self.weight_quant),
            QuantHardTanh(act_quant=self.act_quant),
        )
        self.conv3 = QuantConv2d(self.n_channel,
                                 2 * self.n_channel,
                                 bias=False,
                                 padding=(1, 0),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        self.bn3 = nn.BatchNorm2d(2 * self.n_channel)
        self.pool3 = nn.Sequential(
            QuantConv2d(2 * self.n_channel,
                        2 * self.n_channel,
                        bias=False,
                        stride=(4, 1),
                        kernel_size=(4, 1),
                        weight_quant=self.weight_quant),
            QuantHardTanh(act_quant=self.act_quant),
        )
        # self.conv4 = QuantConv2d(2 * self.n_channel,
        #                          2 * self.n_channel,
        #                          bias=False,
        #                          padding=(1, 0),
        #                          kernel_size=(3, 1),
        #                          weight_quant=self.weight_quant)
        # self.bn4 = nn.BatchNorm2d(2 * self.n_channel)
        # self.pool4 = nn.Sequential(
        #     QuantConv2d(self.n_channel,
        #                 self.n_channel,
        #                 bias=False,
        #                 stride=(4, 1),
        #                 kernel_size=(4, 1),
        #                 weight_quant=self.weight_quant),
        #     QuantHardTanh(act_quant=self.act_quant),
        # )
        self.conv5 = QuantConv2d(2 * self.n_channel,
                                 2 * self.n_channel,
                                 bias=False,
                                 padding=(2, 0),
                                 kernel_size=(3, 1),
                                 weight_quant=self.weight_quant)
        self.bn5 = nn.BatchNorm2d(2 * self.n_channel)
        self.pool5 = nn.Sequential(
            QuantConv2d(2 * self.n_channel,
                        2 * self.n_channel,
                        stride=(4, 1),
                        bias=False,
                        kernel_size=(4, 1),
                        weight_quant=self.weight_quant),
            QuantHardTanh(act_quant=self.act_quant),
        )
        # self.conv6 = QuantConv2d(4 * self.n_channel,
        #                          4 * self.n_channel,
        #                          bias=False,
        #                          padding=(1, 0),
        #                          kernel_size=(3, 1),
        #                          weight_quant=self.weight_quant)
        # self.bn6 = nn.BatchNorm2d(4 * self.n_channel)
        # self.pool6 = nn.Sequential(
        #     QuantConv2d(4 * self.n_channel,
        #                 4 * self.n_channel,
        #                 bias=False,
        #                 stride=(4, 1),
        #                 kernel_size=(4, 1),
        #                 weight_quant=self.weight_quant),
        #     QuantHardTanh(act_quant=self.act_quant),
        # )
        self.fc1 = QuantLinear(2 * self.n_channel,
                               2 * self.n_channel,
                               bias=False,
                               weight_quant=self.weight_quant)
        self.bnfc1 = nn.BatchNorm1d(2 * self.n_channel)
        self.fc2 = QuantLinear(2 * self.n_channel,
                               num_classes,
                               bias=False,
                               weight_quant=self.weight_quant)
        # NOTE: activiation must different instance for
        # MultiThreshol-Add absorption
        self.actpre = QuantHardTanh(act_quant=self.act_quant)
        self.act1 = QuantHardTanh(act_quant=self.act_quant)
        self.act2 = QuantHardTanh(act_quant=self.act_quant)
        self.act3 = QuantHardTanh(act_quant=self.act_quant)
        self.act4 = QuantHardTanh(act_quant=self.act_quant)
        self.act5 = QuantHardTanh(act_quant=self.act_quant)
        self.act6 = QuantHardTanh(act_quant=self.act_quant)
        self.actfc1 = QuantHardTanh(act_quant=self.act_quant)
        self.emb = QuantConv2d(2 * self.n_channel,
                               2 * self.n_channel,
                               bias=False,
                               kernel_size=self.emb_factor,
                               weight_quant=self.weight_quant)

    def forward(self, x):
        x = self.actpre(x)
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
        x = x.view(-1, 2 * self.n_channel)
        if __debug__:
            print(x.shape)
        x = self.fc1(x)
        x = self.actfc1(self.bnfc1(x))
        x = self.fc2(x)
        return x


class M5_NOBN_brevitas(nn.Module):
    def __init__(self,
                 input_channels=1,
                 num_classes=35,
                 stride=4,
                 n_channel=96):
        super().__init__()
        self.emb_factor = (16, 1)
        self.n_channel = n_channel
        self.weight_quant = SignedBinaryWeightPerTensorConst
        self.act_quant = SignedBinaryActPerTensorConst
        self.conv1 = QuantConv2d(input_channels,
                                 self.n_channel,
                                 kernel_size=(84, 1),
                                 stride=stride,
                                 weight_quant=self.weight_quant)
        self.bn1 = QuantIdentity(act_quant=self.act_quant)
        self.pool1 = QuantConv2d(self.n_channel,
                                 self.n_channel,
                                 stride=(4, 1),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        self.conv2 = QuantConv2d(self.n_channel,
                                 self.n_channel,
                                 padding=(2, 0),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        self.bn2 = QuantIdentity(act_quant=self.act_quant)
        self.pool2 = QuantConv2d(self.n_channel,
                                 self.n_channel,
                                 stride=(4, 1),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        self.conv3 = QuantConv2d(self.n_channel,
                                 2 * self.n_channel,
                                 padding=(1, 0),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        self.bn3 = QuantIdentity(act_quant=self.act_quant)
        self.pool3 = QuantConv2d(2 * self.n_channel,
                                 2 * self.n_channel,
                                 stride=(4, 1),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        # self.conv4 = QuantConv2d(2 * self.n_channel,
        #                          2 * self.n_channel,
        #                          padding=(1, 0),
        #                          kernel_size=(3, 1),
        #                          weight_quant=self.weight_quant)
        # self.bn4 = QuantIdentity(act_quant=self.act_quant)
        # self.pool4 = QuantConv2d(self.n_channel,
        #                          self.n_channel,
        #                          stride=(4, 1),
        #                          kernel_size=(4, 1),
        #                          weight_quant=self.weight_quant)
        self.conv5 = QuantConv2d(2 * self.n_channel,
                                 4 * self.n_channel,
                                 padding=(2, 0),
                                 kernel_size=(3, 1),
                                 weight_quant=self.weight_quant)
        self.bn5 = QuantIdentity(act_quant=self.act_quant)
        self.pool5 = QuantConv2d(4 * self.n_channel,
                                 4 * self.n_channel,
                                 stride=(4, 1),
                                 kernel_size=(4, 1),
                                 weight_quant=self.weight_quant)
        # self.conv6 = QuantConv2d(4 * self.n_channel,
        #                          4 * self.n_channel,
        #                          padding=(1, 0),
        #                          kernel_size=(3, 1),
        #                          weight_quant=self.weight_quant)
        # self.bn6 = QuantIdentity(act_quant=self.act_quant)
        # self.pool6 = QuantConv2d(4 * self.n_channel,
        #                          4 * self.n_channel,
        #                          stride=(4, 1),
        #                          kernel_size=(4, 1),
        #                          weight_quant=self.weight_quant)
        self.fc1 = QuantLinear(4 * self.n_channel,
                               2 * self.n_channel,
                               bias=False,
                               weight_quant=self.weight_quant)
        self.bnfc1 = QuantIdentity(act_quant=self.act_quant)
        self.fc2 = QuantLinear(2 * self.n_channel,
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
        self.emb = QuantConv2d(4 * self.n_channel,
                               4 * self.n_channel,
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
        x = x.view(-1, 4 * self.n_channel)
        if __debug__:
            print(x.shape)
        x = self.fc1(x)
        x = self.actfc1(self.bnfc1(x))
        x = self.fc2(x)
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
