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


@register_fx_node_ge_converter(torch.ops.prims.collapse_view.default)
def conveter_prims_collapse_view_default(
    a: Tensor, start: int, end: int, meta_outputs: TensorSpec = None
):
    """NB: prims::collapse_view(Tensor(a) a, int start, int end) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.prims.collapse_view.default ge_converter is not implemented!")
