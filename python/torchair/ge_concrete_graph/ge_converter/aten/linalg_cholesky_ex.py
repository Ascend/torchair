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


@register_fx_node_ge_converter(torch.ops.aten.linalg_cholesky_ex.default)
def conveter_aten_linalg_cholesky_ex_default(
        self: Tensor,
        *,
        upper: bool = False,
        check_errors: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info) """
    raise NotImplementedError("torch.ops.aten.linalg_cholesky_ex.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_cholesky_ex.L)
def conveter_aten_linalg_cholesky_ex_L(
        self: Tensor,
        *,
        upper: bool = False,
        check_errors: bool = False,
        L: Tensor = None,
        info: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::linalg_cholesky_ex.L(Tensor self, *, bool upper=False, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info) """
    raise NotImplementedError("torch.ops.aten.linalg_cholesky_ex.L ge converter is not implement!")


