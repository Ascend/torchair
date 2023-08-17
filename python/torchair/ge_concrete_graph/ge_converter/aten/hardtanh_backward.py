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
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2), F32(2, 2), -1., 1.),
    Support(F32(2, 2), F32(2, 2), -2., 2.),
    Support(F16(2, 2), F16(2, 2), -1., 1.),
])
@register_fx_node_ge_converter(torch.ops.aten.hardtanh_backward.default)
def conveter_aten_hardtanh_backward_default(
    grad_output: Tensor,
    self: Tensor,
    min_val: Union[Number, Tensor],
    max_val: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor"""
    if isinstance(min_val, Tensor) or isinstance(max_val, Tensor):
        raise NotImplementedError("torch.ops.aten.hardtanh_backward.default ge_converter is not implemented "
                                  "when min_val or max_val is tensor!")
    return ge.HardtanhGrad(self, grad_output, min_val=min_val, max_val=max_val)


@register_fx_node_ge_converter(torch.ops.aten.hardtanh_backward.grad_input)
def conveter_aten_hardtanh_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    min_val: Union[Number, Tensor],
    max_val: Union[Number, Tensor],
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.hardtanh_backward.grad_input ge_converter is not implemented!")
