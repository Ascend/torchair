import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
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


@register_fx_node_ge_converter(torch.ops.quantized.instance_norm.default)
def conveter_quantized_instance_norm_default(
        input: Tensor,
        weight: Optional[Tensor],
        bias: Optional[Tensor],
        eps: float,
        output_scale: float,
        output_zero_point: int,
        meta_outputs: Any = None):
    """ NB: quantized::instance_norm(Tensor input, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> Tensor """
    raise NotImplementedError("torch.ops.quantized.instance_norm.default ge converter is not implement!")


