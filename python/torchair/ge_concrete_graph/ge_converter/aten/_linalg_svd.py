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


@register_fx_node_ge_converter(torch.ops.aten._linalg_svd.default)
def conveter_aten__linalg_svd_default(
        A: Tensor,
        full_matrices: bool = False,
        compute_uv: bool = True,
        *,
        driver: Optional[str] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_linalg_svd(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh) """
    raise NotImplementedError("torch.ops.aten._linalg_svd.default ge converter is not implement!")


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
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_linalg_svd.U(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) """
    raise NotImplementedError("torch.ops.aten._linalg_svd.U ge converter is not implement!")


