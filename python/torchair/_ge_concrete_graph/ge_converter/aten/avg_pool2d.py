from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, assert_args_checkout
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
     Support
from torchair._ge_concrete_graph.utils import specific_op_input_layout, specific_op_output_layout


@declare_supported([
     Support(F32(16, 16, 4, 4), [2, 2], [1, 1], [0, 0], True, False),
     Support(F32(16, 16, 4, 4), [1, 1], [2, 2], [0, 0], False, False),
     Support(F32(96, 16, 4, 4), [4, 4]),
     Support(F32(16, 4, 4), [4, 4])
])
@register_fx_node_ge_converter(torch.ops.aten.avg_pool2d.default)
def conveter_aten_avg_pool2d_default(
    self: Tensor,
    kernel_size: List[int],
    stride: List[int] = [],
    padding: List[int] = [0, 0],
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor"""
    assert_args_checkout(len(kernel_size) == 1 or len(kernel_size) == 2,
                         "torch.ops.aten.avg_pool2d.default: kernel_size must either be a single int, or a tuple of "
                         "two ints")
    assert_args_checkout(len(stride) == 0 or len(stride) == 1 or len(stride) == 2,
                         "torch.ops.aten.avg_pool2d.default: stride must either be omitted, a single int, or a tuple "
                         "of two ints")
    assert_args_checkout(len(padding) == 1 or len(padding) == 2,
                         "torch.ops.aten.avg_pool2d.default: padding must either be a single int, or a tuple of two "
                         "ints")
    assert_args_checkout(self.rank == 3 or self.rank == 4,
                                 "non-empty 3D or 4D (batch mode) tensor expected for input")
    assert_args_checkout(divisor_override is None or (0 < divisor_override <= 255),
                 f"The value of divisor_override = {divisor_override} is invalid, only support [1, 255] at present.")
    self_copy = self
    if self.rank == 3:
        self_copy = ge.Unsqueeze(self_copy, axes=[0])
    k_h = kernel_size[0]
    k_w = k_h if len(kernel_size) == 1 else kernel_size[1]
    kernel_sizes = [k_h, k_w]
    d_h = k_h if len(stride) == 0 else stride[0]
    d_w = k_w if len(stride) == 0 else d_h if len(stride) == 1 else stride[1]
    strides = [d_h, d_w]
    pad_h = padding[0]
    pad_w = pad_h if len(padding) == 1 else padding[1]
    paddings = [pad_h, pad_w]
    kernel_size_new = [1, 1, kernel_sizes[0], kernel_sizes[1]]
    strides_size_new = [1, 1, strides[0], strides[1]]
    pads = [paddings[0], paddings[0], paddings[1], paddings[1]]
    divisor_override_value = 0
    if divisor_override:
        divisor_override_value = divisor_override
    if self.dtype == DataType.DT_FLOAT16 or self.dtype == DataType.DT_INT8:
        result = ge.AvgPoolV2(self_copy, ksize=kernel_size_new, strides=strides_size_new, pads=pads,
                              ceil_mode=ceil_mode, exclusive=True, divisor_override=divisor_override_value)
        specific_op_input_layout(result, indices=[0], layout="NCHW")
        specific_op_output_layout(result, indices=0, layout="NCHW")
    else:
        result = ge.AvgPoolV2(self_copy, ksize=kernel_size_new, strides=strides_size_new, pads=pads,
                              ceil_mode=ceil_mode, exclusive=not count_include_pad, 
                              divisor_override=divisor_override_value)
        specific_op_input_layout(result, indices=[0], layout="NCHW")
        specific_op_output_layout(result, indices=0, layout="NCHW")
    if self.rank == 3:
        result = ge.Squeeze(result, axis=[0])
    return result


@register_fx_node_ge_converter(torch.ops.aten.avg_pool2d.out)
def conveter_aten_avg_pool2d_out(
    self: Tensor,
    kernel_size: List[int],
    stride: List[int] = [],
    padding: List[int] = [0, 0],
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.avg_pool2d.out ge_converter is not implemented!")
