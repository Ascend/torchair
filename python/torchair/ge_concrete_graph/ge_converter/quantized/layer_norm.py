import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.quantized.layer_norm.default)
def conveter_quantized_layer_norm_default(
        input: Tensor,
        normalized_shape: List[int],
        weight: Optional[Tensor],
        bias: Optional[Tensor],
        eps: float,
        output_scale: float,
        output_zero_point: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: quantized::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> Tensor """
    raise NotImplementedError("torch.ops.quantized.layer_norm.default ge converter is not implement!")


