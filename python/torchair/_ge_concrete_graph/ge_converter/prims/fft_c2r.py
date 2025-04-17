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


@register_fx_node_ge_converter(torch.ops.prims.fft_c2r.default)
def conveter_prims_fft_c2r_default(
    self: Tensor,
    *,
    dim: List[int],
    last_dim_size: Union[int, Tensor],
    meta_outputs: TensorSpec = None
):
    """NB: prims::fft_c2r(Tensor self, *, int[] dim, SymInt last_dim_size) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.fft_c2r.default ge_converter is not implemented!")
