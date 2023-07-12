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


@register_fx_node_ge_converter(torch.ops.quantized.leaky_relu.default)
def conveter_quantized_leaky_relu_default(
        qx: Tensor,
        negative_slope: Union[Number, Tensor],
        inplace: bool,
        output_scale: float,
        output_zero_point: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: quantized::leaky_relu(Tensor qx, Scalar negative_slope, bool inplace, float output_scale, int output_zero_point) -> Tensor """
    raise NotImplementedError("torch.ops.quantized.leaky_relu.default ge converter is not implement!")


