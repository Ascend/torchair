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


@register_fx_node_ge_converter(torch.ops.aten.linalg_inv_ex.default)
def conveter_aten_linalg_inv_ex_default(
    A: Tensor, *, check_errors: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_inv_ex(Tensor A, *, bool check_errors=False) -> (Tensor inverse, Tensor info)"""
    raise NotImplementedError("torch.ops.aten.linalg_inv_ex.default Ascend op converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_inv_ex.inverse)
def conveter_aten_linalg_inv_ex_inverse(
    A: Tensor,
    *,
    check_errors: bool = False,
    inverse: Tensor = None,
    info: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linalg_inv_ex.inverse(Tensor A, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info)"""
    raise NotImplementedError("torch.ops.aten.linalg_inv_ex.inverse ge_converter is not implemented!")
