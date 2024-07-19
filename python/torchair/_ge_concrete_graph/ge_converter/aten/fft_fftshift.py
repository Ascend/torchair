from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.fft_fftshift.default)
def conveter_aten_fft_fftshift_default(
    self: Tensor, dim: Optional[List[int]] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::fft_fftshift(Tensor self, int[1]? dim=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.fft_fftshift.default ge_converter is not implemented!")
