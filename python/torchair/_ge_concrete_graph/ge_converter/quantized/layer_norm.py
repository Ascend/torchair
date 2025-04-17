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


@register_fx_node_ge_converter(torch.ops.quantized.layer_norm.default)
def conveter_quantized_layer_norm_default(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    output_scale: float,
    output_zero_point: int,
    meta_outputs: TensorSpec = None,
):
    """NB: quantized::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> Tensor"""
    raise NotImplementedError("torch.ops.quantized.layer_norm.default ge_converter is not implemented!")
