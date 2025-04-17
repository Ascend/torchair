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


@register_fx_node_ge_converter(torch.ops.aten._linalg_svd.default)
def conveter_aten__linalg_svd_default(
    A: Tensor,
    full_matrices: bool = False,
    compute_uv: bool = True,
    *,
    driver: Optional[str] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_linalg_svd(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)"""
    raise NotImplementedError("torch.ops.aten._linalg_svd.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._linalg_svd.U)
def conveter_aten__linalg_svd_U(
    A: Tensor,
    full_matrices: bool = False,
    compute_uv: bool = True,
    *,
    driver: Optional[str] = None,
    U: Tensor = None,
    S: Tensor = None,
    Vh: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_linalg_svd.U(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)"""
    raise NotImplementedError("torch.ops.aten._linalg_svd.U ge_converter is not implemented!")
