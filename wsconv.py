from typing import Union, Tuple, Type, Optional
import math

import torch
from torch.nn import Parameter, Module, init
from torch import Tensor

from brevitas.nn import QuantConv2d
from brevitas.quant_tensor import QuantTensor
from brevitas.inject.defaults import Int8WeightPerTensorFloat
from brevitas.nn.quant_layer import WeightQuantType, BiasQuantType, ActQuantType

__all__ = ['NegBiasLayer', 'WSConv2d']

class NegBiasLayer(Module):
    def __init__(self, *args, device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NegBiasLayer, self).__init__()
        self.in_features = 1
        self.out_features = 1
        self.bias = Parameter(
            torch.empty(self.in_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            return input - torch.abs(self.bias)
        else:
            return input - torch.abs(self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class WSConv2d(QuantConv2d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_type: str = 'standard',
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        QuantConv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)

    def forward(self, input: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:

        layer_std, layer_mean = torch.std_mean(self.weight.data)
        self.weight.data = self.weight.data - layer_mean
        self.weight.data = self.weight.data / layer_std
        self.weight.data = self.weight.data * torch.numel(self.weight.data)**-.5
        return self.forward_impl(input)
