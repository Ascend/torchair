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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, torch_type_to_ge_type, \
    declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote

@declare_supported([
    Support(F32(96, 512, 1, 1), F32(96, 512, 4, 4), [4, 4], [], [0, 0], False, True, divisor_override=None),
    Support(F32(96, 512, 1, 1), F32(96, 512, 4, 4), [4], [], [0, 0], False, True, divisor_override=None),
    Support(F32(96, 512, 1, 1), F32(96, 512, 4, 4), [4, 4], [1], [0], False, True, divisor_override=None),
])
@register_fx_node_ge_converter(torch.ops.aten.avg_pool2d_backward.default)
def conveter_aten_avg_pool2d_backward_default(
    grad_output: Tensor,
    self: Tensor,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor"""
    k_h = kernel_size[0]
    k_w = k_h if len(kernel_size) == 1 else kernel_size[1]
    kernel_sizes = [k_h, k_w]
    d_h = k_h if len(stride) == 0 else stride[0]
    d_w = k_w if len(stride) == 0 else d_h if len(stride) == 1 else stride[1]
    strides = [d_h, d_w]
    pad_h = padding[0]
    pad_w = pad_h if len(padding) == 1 else padding[1]
    paddings = [pad_h, pad_w]
    kernel_size = [1, 1, kernel_sizes[0], kernel_sizes[1]]
    strides_size = [1, 1, strides[0], strides[1]]
    pads = [paddings[0], paddings[0], paddings[1], paddings[1]]
    exclusive = False if count_include_pad else True

    output = ge.AvgPoolV2Grad(ge.Shape(self), grad_output, ksize=kernel_size, strides=strides_size,
                              padding_mode="CALCULATED", pads=pads, data_format="NCHW",
                              global_pooling=False, ceil_mode=ceil_mode, exclusive=exclusive)
    output._node.input_desc[1].layout = "NCHW"
    output._node.output_desc[0].layout = "NCHW"
    return output

@register_fx_node_ge_converter(torch.ops.aten.avg_pool2d_backward.grad_input)
def conveter_aten_avg_pool2d_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise AssertionError("torch.ops.aten.avg_pool2d_backward.grad_input is redundant before pytorch 2.1.0, "
                         "might be supported in future version.")
