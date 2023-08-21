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


@register_fx_node_ge_converter(torch.ops.aten.native_group_norm_backward.default)
def conveter_aten_native_group_norm_backward_default(
    grad_out: Tensor,
    input: Tensor,
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    N: Union[int, Tensor],
    C: Union[int, Tensor],
    HxW: Union[int, Tensor],
    group: int,
    output_mask: List[bool],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.native_group_norm_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.native_group_norm_backward.out)
def conveter_aten_native_group_norm_backward_out(
    grad_out: Tensor,
    input: Tensor,
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    N: Union[int, Tensor],
    C: Union[int, Tensor],
    HxW: Union[int, Tensor],
    group: int,
    output_mask: List[bool],
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    out2: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::native_group_norm_backward.out(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise NotImplementedError("torch.ops.aten.native_group_norm_backward.out ge_converter is not implemented!")
