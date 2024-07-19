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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2, 4), F32(2, 2, 2), [1, 1, 1, 1]),
    Support(F32(2, 2, 4, 4), F32(2, 2, 2, 2), [1, 1, 1, 1]),
    Support(F32(2, 2, 3, 4), F32(2, 2, 2, 2), [1, 1, 0, 1]),
])
@register_fx_node_ge_converter(torch.ops.aten.reflection_pad2d_backward.default)
def conveter_aten_reflection_pad2d_backward_default(
    grad_output: Tensor,
    self: Tensor,
    padding: Union[List[int], Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, SymInt[4] padding) -> Tensor"""
    self_rank = self.rank
    if self.rank == 3:
        self_rank = 4
        grad_output = ge.Unsqueeze(grad_output, axes=[0])

    # padding padding to all dims
    padding.extend([0 for _ in range(2 * self_rank - len(padding))])

    # convert torch padding to PadV3Grad padding required, WHCN -> NCHW
    padding.reverse()
    for i in range(len(padding) // 2):
        tmp = padding[i * 2]
        padding[i * 2] = padding[i * 2 + 1]
        padding[i * 2 + 1] = tmp

    output = ge.PadV3Grad(grad_output, padding, mode='reflect', paddings_contiguous=True)

    if self.rank == 3:
        output = ge.Squeeze(output, axis=[0])

    return output


@register_fx_node_ge_converter(torch.ops.aten.reflection_pad2d_backward.grad_input)
def conveter_aten_reflection_pad2d_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    padding: Union[List[int], Tensor],
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::reflection_pad2d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.reflection_pad2d_backward.grad_input ge_converter is not supported!")
