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


@register_fx_node_ge_converter(torch.ops.aten.linalg_inv_ex.default)
def conveter_aten_linalg_inv_ex_default(
        A: Tensor,
        *,
        check_errors: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::linalg_inv_ex(Tensor A, *, bool check_errors=False) -> (Tensor inverse, Tensor info) """
    raise NotImplementedError("torch.ops.aten.linalg_inv_ex.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_inv_ex.inverse)
def conveter_aten_linalg_inv_ex_inverse(
        A: Tensor,
        *,
        check_errors: bool = False,
        inverse: Tensor = None,
        info: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::linalg_inv_ex.inverse(Tensor A, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info) """
    raise NotImplementedError("torch.ops.aten.linalg_inv_ex.inverse ge converter is not implement!")


