import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
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


@register_fx_node_ge_converter(torch.ops.aten._linalg_eigh.default)
def conveter_aten__linalg_eigh_default(
        A: Tensor,
        UPLO: str = "L",
        compute_v: bool = True,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_linalg_eigh(Tensor A, str UPLO="L", bool compute_v=True) -> (Tensor eigenvalues, Tensor eigenvectors) """
    raise NotImplementedError("torch.ops.aten._linalg_eigh.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._linalg_eigh.eigenvalues)
def conveter_aten__linalg_eigh_eigenvalues(
        A: Tensor,
        UPLO: str = "L",
        compute_v: bool = True,
        *,
        eigenvalues: Tensor = None,
        eigenvectors: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_linalg_eigh.eigenvalues(Tensor A, str UPLO="L", bool compute_v=True, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) """
    raise NotImplementedError("torch.ops.aten._linalg_eigh.eigenvalues ge converter is not implement!")


