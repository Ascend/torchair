from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
import numpy as np
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair.ge._ge_graph import compat_as_bytes


# No testcase because the dtype and shape of input *mask* are different from cpu's.
@register_fx_node_ge_converter(torch.ops.aten.native_dropout_backward.default)
def conveter_aten_native_dropout_backward_default(
    grad_output: Tensor, mask: Tensor, scale: float, meta_outputs: TensorSpec = None
):
    """NB: aten::native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor"""
    retain = 1. / scale
    return ge.DropOutDoMask(grad_output, mask, retain)


@register_fx_node_ge_converter(torch.ops.aten.native_dropout_backward.out)
def conveter_aten_native_dropout_backward_out(
    grad_output: Tensor,
    mask: Tensor,
    scale: float,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::native_dropout_backward.out(Tensor grad_output, Tensor mask, float scale, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.native_dropout_backward.out ge_converter is not implemented!")
