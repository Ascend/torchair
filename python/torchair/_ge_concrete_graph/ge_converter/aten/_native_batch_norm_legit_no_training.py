from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.utils import specific_op_input_layout, \
    specific_op_output_layout
from torchair._ge_concrete_graph.supported_declaration import F32, Support
from torchair._utils.check_platform import is_not_support


@declare_supported(
    [
        Support(F32(2, 2, 2), F32(2), F32(2), F32(2), F32(2), momentum=0.9, eps=1e-5),
        Support(F32(2, 2, 2, 2), F32(2), F32(2), F32(2), F32(2), momentum=0.9, eps=1e-5),
        Support(F32(2, 2, 2, 2, 2), F32(2), F32(2), F32(2), F32(2), momentum=0.9, eps=1e-5),
        Support(F32(2, 2, 2, 2), None, None, F32(2), F32(2), momentum=0.9, eps=1e-5),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit_no_training.default)
def conveter_aten__native_batch_norm_legit_no_training_default(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)"""
    dim = input.rank
    if dim < 2:
        raise RuntimeError("torch.ops.aten._native_batch_norm_legit_no_training.default "
                           "ge_converter is not implemented while input dim <2!")
    input_size = ge.Shape(input, dtype=DataType.DT_INT32)
    if weight is None:
        weight = ge.Fill(ge.Gather(input_size, 1), ge.Cast(1., dst_type=input.dtype))
        
    if bias is None:
        bias = ge.Fill(ge.Gather(input_size, 1), ge.Cast(0., dst_type=input.dtype))
    empty = ge.Fill([0], ge.Cast(0.0, dst_type=input.dtype))

    if is_not_support():
        if dim == 2 or dim == 3:
            input = ge.Unsqueeze(input, axes=list(range(dim, 4)))

        if dim > 5:
            shape_list = ge.Pack([ge.Gather(input_size, 0), ge.Gather(input_size, 1), ge.Gather(input_size, 2),
                                  ge.Gather(input_size, 3), -1], N=5, axis=0)
            input = ge.Reshape(input, shape_list)

        output, _, _, _, _ = ge.BatchNormV3(input, weight, bias, running_mean, running_var,
                                            epsilon=eps, momentum=momentum, is_training=False)
        if dim <= 4:
            specific_op_input_layout(output, indices=0, layout="NCHW")
            specific_op_output_layout(output, indices=0, layout="NCHW")
            if dim == 2 or dim == 3:
                output = ge.Squeeze(output, axis=list(range(3, dim - 1, -1)))
        else:
            specific_op_input_layout(output, indices=0, layout="NCDHW")
            specific_op_output_layout(output, indices=0, layout="NCDHW")
            if dim > 5:
                output = ge.Reshape(output, input_size)
                specific_op_input_layout(output, indices=[0, 1], layout="ND")
                specific_op_output_layout(output, indices=0, layout="ND")
        return output, empty, empty
    
    
    if dim == 2 or dim == 3:
        input = ge.Unsqueeze(input, axes=list(range(dim, 4)))

    if dim >= 5:
        shape_list = ge.Pack([ge.Gather(input_size, 0), ge.Gather(input_size, 1),
                              ge.Gather(input_size, 2), -1], N=4, axis=0)
        input = ge.Reshape(input, shape_list)
    
    output = ge.BNInfer(input, weight, bias, running_mean, running_var, epsilon=eps)
    specific_op_input_layout(output, indices=list(range(5)), layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    if dim == 2 or dim == 3:
        output = ge.Squeeze(output, axis=list(range(3, dim - 1, -1)))

    if dim >= 5:
        output = ge.Reshape(output, input_size)
        specific_op_input_layout(output, indices=[0, 1], layout="ND")
        specific_op_output_layout(output, indices=0, layout="ND")

    return output, empty, empty


@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit_no_training.out)
def conveter_aten__native_batch_norm_legit_no_training_out(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    out2: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_native_batch_norm_legit_no_training.out(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise RuntimeError("torch.ops.aten._native_batch_norm_legit_no_training.out ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")
