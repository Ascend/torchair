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


@register_fx_node_ge_converter(torch.ops.aten._linalg_eigh.default)
def conveter_aten__linalg_eigh_default(
    A: Tensor, UPLO: str = "L", compute_v: bool = True, meta_outputs: TensorSpec = None
):
    """NB: aten::_linalg_eigh(Tensor A, str UPLO="L", bool compute_v=True) -> (Tensor eigenvalues, Tensor eigenvectors)"""
    raise NotImplementedError("torch.ops.aten._linalg_eigh.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._linalg_eigh.eigenvalues)
def conveter_aten__linalg_eigh_eigenvalues(
    A: Tensor,
    UPLO: str = "L",
    compute_v: bool = True,
    *,
    eigenvalues: Tensor = None,
    eigenvectors: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_linalg_eigh.eigenvalues(Tensor A, str UPLO="L", bool compute_v=True, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)"""
    raise NotImplementedError("torch.ops.aten._linalg_eigh.eigenvalues ge_converter is not implemented!")
