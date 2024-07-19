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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.utils import specific_op_input_layout, \
    specific_op_output_layout
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support


@declare_supported(
    [
        Support(F32(2, 2, 2), F32(2), F32(2), F32(2), F32(2), training=True, momentum=0.9, eps=1e-5),
        Support(F32(2, 2, 2, 2), F32(2), F32(2), F32(2), F32(2), training=True, momentum=0.9, eps=1e-5),
        Support(F32(2, 2, 2, 2, 2), F32(2), F32(2), F32(2), F32(2), training=True, momentum=0.9, eps=1e-5),
        Support(F32(2, 2, 2, 2), None, None, F32(2), F32(2), training=True, momentum=0.9, eps=1e-5),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit.default)
def conveter_aten__native_batch_norm_legit_default(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)"""
    if not training:
        raise RuntimeError(
            "torch.ops.aten._native_batch_norm_legit.default ge_converter is not implemented while training is False!"
        )
    dim = input.rank
    if dim > 5 or dim < 2:
        raise RuntimeError("torch.ops.aten._native_batch_norm_legit.default "
                           "ge_converter is not implemented while input dim > 5 or input dim <2!")
    input_size = ge.Shape(input, dtype=DataType.DT_INT32)
    if weight is None:
        weight = ge.Fill(ge.Gather(input_size, 1), ge.Cast(1., dst_type=input.dtype))

    if bias is None:
        bias = ge.Fill(ge.Gather(input_size, 1), ge.Cast(0., dst_type=input.dtype))

    if dim <= 4:
        if dim == 2 or dim == 3:
            input = ge.Unsqueeze(input, axes=list(range(dim, 4)))
        sum_output, square_sum = ge.BNTrainingReduce(input)
        specific_op_input_layout(sum_output, indices=0, layout="NCHW")
        specific_op_output_layout(sum_output, indices=[0, 1], layout="NCHW")
        output, mean, variance, batch_mean, batch_variance = ge.BNTrainingUpdate(input, sum_output, \
                                                                                 square_sum, weight, bias, running_mean,
                                                                                 running_var, factor=momentum,
                                                                                 epsilon=eps)
        specific_op_input_layout(output, indices=list(range(7)), layout="NCHW")
        specific_op_output_layout(output, indices=list(range(5)), layout="NCHW")
        if dim == 2 or dim == 3:
            output = ge.Squeeze(output, axis=list(range(3, dim - 1, -1)))
        result = (output, batch_mean, batch_variance, mean, variance)
        return result

    sum_output, square_sum = ge.BN3DTrainingReduce(input)
    specific_op_input_layout(sum_output, indices=0, layout="NCDHW")
    specific_op_output_layout(sum_output, indices=[0, 1], layout="NCDHW")
    output, mean, variance, batch_mean, batch_variance = ge.BN3DTrainingUpdate(input, sum_output, \
                                                                               square_sum, weight, bias, running_mean,
                                                                               running_var, factor=momentum,
                                                                               epsilon=eps)
    specific_op_input_layout(output, indices=list(range(7)), layout="NCDHW")
    specific_op_output_layout(output, indices=list(range(5)), layout="NCDHW")
    result = (output, batch_mean, batch_variance, mean, variance)
    return result


@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit.no_stats)
def conveter_aten__native_batch_norm_legit_no_stats(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)"""
    if not training:
        raise RuntimeError(
            "torch.ops.aten._native_batch_norm_legit.no_stats ge_converter is not implemented while training is False!"
        )
    dim = input.rank
    if dim > 5 or dim < 2:
        raise RuntimeError("torch.ops.aten._native_batch_norm_legit.no_stats "
                           "ge_converter is not implemented while input dim > 5 or input dim <2!")
    input_size = ge.Shape(input, dtype=DataType.DT_INT32)
    if weight is None:
        weight = ge.Fill(ge.Gather(input_size, 1), ge.Cast(1., dst_type=input.dtype))

    if bias is None:
        bias = ge.Fill(ge.Gather(input_size, 1), ge.Cast(0., dst_type=input.dtype))

    running_mean = ge.Fill(ge.Gather(input_size, 1), ge.Cast(0., dst_type=input.dtype))
    running_var = ge.Fill(ge.Gather(input_size, 1), ge.Cast(1., dst_type=input.dtype))
    if dim <= 4:
        if dim == 2 or dim == 3:
            input = ge.Unsqueeze(input, axes=list(range(dim, 4)))
        sum_output, square_sum = ge.BNTrainingReduce(input)
        specific_op_input_layout(sum_output, indices=0, layout="NCHW")
        specific_op_output_layout(sum_output, indices=[0, 1], layout="NCHW")
        output, mean, variance, batch_mean, batch_variance = ge.BNTrainingUpdate(input, sum_output,
                                                                                 square_sum, weight, bias, running_mean,
                                                                                 running_var, factor=momentum,
                                                                                 epsilon=eps)
        specific_op_input_layout(output, indices=list(range(7)), layout="NCHW")
        specific_op_output_layout(output, indices=list(range(5)), layout="NCHW")
        if dim == 2 or dim == 3:
            output = ge.Squeeze(output, axis=list(range(3, dim - 1, -1)))
        result = (output, batch_mean, batch_variance)
        return result

    sum_output, square_sum = ge.BN3DTrainingReduce(input)
    specific_op_input_layout(sum_output, indices=0, layout="NCDHW")
    specific_op_output_layout(sum_output, indices=[0, 1], layout="NCDHW")
    output, mean, variance, batch_mean, batch_variance = ge.BN3DTrainingUpdate(input, sum_output,
                                                                               square_sum, weight, bias, running_mean,
                                                                               running_var, factor=momentum,
                                                                               epsilon=eps)
    specific_op_input_layout(output, indices=list(range(7)), layout="NCDHW")
    specific_op_output_layout(output, indices=list(range(5)), layout="NCDHW")
    result = (output, batch_mean, batch_variance)
    return result


@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit.out)
def conveter_aten__native_batch_norm_legit_out(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
    *,
    out: Tensor = None,
    save_mean: Tensor = None,
    save_invstd: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_native_batch_norm_legit.out(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps, *, Tensor(d!) out, Tensor(e!) save_mean, Tensor(f!) save_invstd) -> (Tensor(d!), Tensor(e!), Tensor(f!))"""
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit.no_stats_out)
def conveter_aten__native_batch_norm_legit_no_stats_out(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
    *,
    out: Tensor = None,
    save_mean: Tensor = None,
    save_invstd: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_native_batch_norm_legit.no_stats_out(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit.no_stats_out ge_converter is not implemented!")
