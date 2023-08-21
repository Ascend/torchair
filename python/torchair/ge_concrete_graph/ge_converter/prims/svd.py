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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.prims.svd.default)
def conveter_prims_svd_default(
    A: Tensor, *, full_matrices: bool, meta_outputs: TensorSpec = None
):
    """NB: prims::svd(Tensor A, *, bool full_matrices) -> (Tensor U, Tensor S, Tensor Vh)"""
    raise NotImplementedError("torch.ops.prims.svd.default ge_converter is not implemented!")
