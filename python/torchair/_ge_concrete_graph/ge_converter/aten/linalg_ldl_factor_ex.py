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


@register_fx_node_ge_converter(torch.ops.aten.linalg_ldl_factor_ex.default)
def conveter_aten_linalg_ldl_factor_ex_default(
    self: Tensor,
    *,
    hermitian: bool = False,
    check_errors: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_ldl_factor_ex(Tensor self, *, bool hermitian=False, bool check_errors=False) -> (Tensor LD, Tensor pivots, Tensor info)"""
    raise NotImplementedError("torch.ops.aten.linalg_ldl_factor_ex.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_ldl_factor_ex.out)
def conveter_aten_linalg_ldl_factor_ex_out(
    self: Tensor,
    *,
    hermitian: bool = False,
    check_errors: bool = False,
    LD: Tensor = None,
    pivots: Tensor = None,
    info: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_ldl_factor_ex.out(Tensor self, *, bool hermitian=False, bool check_errors=False, Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info)"""
    raise NotImplementedError("torch.ops.aten.linalg_ldl_factor_ex.out ge_converter is not implemented!")
