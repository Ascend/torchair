from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.quantized.dropout.default)
def conveter_quantized_dropout_default(
    self: Tensor,
    output_scale: float,
    output_zero_point: int,
    p: Union[Number, Tensor] = 0.5,
    training: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: quantized::dropout(Tensor self, float output_scale, int output_zero_point, Scalar p=0.5, bool training=False) -> Tensor"""
    raise NotImplementedError("torch.ops.quantized.dropout.default ge_converter is not implemented!")
