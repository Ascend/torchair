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


@register_fx_node_ge_converter(torch.ops.aten.linalg_ldl_factor_ex.default)
def conveter_aten_linalg_ldl_factor_ex_default(
        self: Tensor,
        *,
        hermitian: bool = False,
        check_errors: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::linalg_ldl_factor_ex(Tensor self, *, bool hermitian=False, bool check_errors=False) -> (Tensor LD, Tensor pivots, Tensor info) """
    raise NotImplementedError("torch.ops.aten.linalg_ldl_factor_ex.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_ldl_factor_ex.out)
def conveter_aten_linalg_ldl_factor_ex_out(
        self: Tensor,
        *,
        hermitian: bool = False,
        check_errors: bool = False,
        LD: Tensor = None,
        pivots: Tensor = None,
        info: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::linalg_ldl_factor_ex.out(Tensor self, *, bool hermitian=False, bool check_errors=False, Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info) """
    raise NotImplementedError("torch.ops.aten.linalg_ldl_factor_ex.out ge converter is not implement!")


