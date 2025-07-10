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
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support
from torchair._utils.check_platform import is_not_support


@declare_supported(
    [
        Support(F32(2, 2, 2), F32(2, 2, 2), F32(2), F32(2), F32(2), F32(2), F32(2),
                train=True, eps=1e-5, output_mask=[True, True, True]),
        Support(F32(2, 2, 2, 2), F32(2, 2, 2, 2), F32(2), F32(2), F32(2), F32(2), F32(2),
                train=True, eps=1e-5, output_mask=[True, True, True]),
        Support(F32(2, 2, 2, 2, 2), F32(2, 2, 2, 2, 2), F32(2), F32(2), F32(2), F32(2), F32(2),
                train=True, eps=1e-5, output_mask=[True, True, True]),
        Support(F32(2, 2, 2, 2, 2), F32(2, 2, 2, 2, 2), None, F32(2), F32(2), F32(2), F32(2),
                train=True, eps=1e-5, output_mask=[True, True, True]),
        Support(F32(96, 64, 64, 64), F32(96, 64, 64, 64), F32(64,), F32(64,), F32(64,),
                F32(64,), F32(64,), False, 1e-05, [True, True, True]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.native_batch_norm_backward.default)
def conveter_aten_native_batch_norm_backward_default(
    grad_out: Tensor,
    input: Tensor,
    weight: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_invstd: Optional[Tensor],
    train: bool,
    eps: float,
    output_mask: List[bool],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""
    dim = input.rank
    if dim < 2:
        raise RuntimeError("torch.ops.aten.native_batch_norm_backward.default "
                           "ge_converter is not implemented while input dim <2!")
    input_size = ge.Shape(input, dtype=DataType.DT_INT32)
    if weight is None:
        weight = ge.Fill(ge.Gather(input_size, 1), ge.Cast(1., dst_type=input.dtype))
    
    if is_not_support():
        if dim <= 4:
            grad_in, diff_scale, diff_offset = \
                __native_batch_norm_backward_regbase(dim, grad_out, input, weight, running_mean, running_var,
                                                     save_mean, save_invstd, train, eps, output_mask, meta_outputs)
        else:
            grad_in, diff_scale, diff_offset = \
                __native_batch_norm_backward_regbase_3d(dim, input_size, grad_out, input, weight, running_mean,
                                                        running_var, save_mean, save_invstd, train, eps,
                                                        output_mask, meta_outputs)
    else:
        if train:
            mean = save_mean
            invstd = save_invstd
        else:
            mean = running_mean
            invstd = running_var

        if dim <= 4:
            grad_in, diff_scale, diff_offset = \
                __native_batch_norm_backward(dim, eps, grad_out, invstd, mean, output_mask, train, weight, input)
        else:
            grad_in, diff_scale, diff_offset = \
                __native_batch_norm_backward_3d(dim, input_size, eps, grad_out, invstd, mean, output_mask,
                                                train, weight, input)

    return __mask_output(grad_in, diff_scale, diff_offset, output_mask)


def __compute_scale(dim_list, dy_casted, x_casted, mean_casted, var_casted, eps, layout):
    var1 = ge.Add(var_casted, ge.Cast(eps, dst_type=DataType.DT_FLOAT))
    invstd1 = ge.Pow(var1, ge.Cast(-0.5, dst_type=DataType.DT_FLOAT))
    if layout == "NCHW":
        unsqu_mean = ge.Unsqueeze(mean_casted, axes=[1, 2])
        unsqu_invstd = ge.Unsqueeze(invstd1, axes=[1, 2])
    if layout == "NCDHW":
        unsqu_mean = ge.Unsqueeze(mean_casted, axes=[1, 2, 3])
        unsqu_invstd = ge.Unsqueeze(invstd1, axes=[1, 2, 3])

    x_hat = ge.Sub(x_casted, unsqu_mean)
    x_hat_mul = ge.Mul(x_hat, unsqu_invstd)
    mul_ret = ge.Mul(dy_casted, x_hat_mul)
    sum_ret = ge.ReduceSum(mul_ret, axes=dim_list, keep_dims=False)
    return sum_ret


def __calc_infer_output(grad_out, x, running_mean, running_var, eps, output_mask,
                        diff_scale, diff_offset, layout, meta_outputs):
    if not output_mask[1] and not output_mask[2]:
        return diff_scale, diff_offset
    dy_casted = ge.Cast(grad_out, dst_type=DataType.DT_FLOAT)
    x_casted = ge.Cast(x, dst_type=DataType.DT_FLOAT)
    mean_casted = ge.Cast(running_mean, dst_type=DataType.DT_FLOAT)
    var_casted = ge.Cast(running_var, dst_type=DataType.DT_FLOAT)

    dim_list = []
    if layout == "NCHW":
        dim_list = [0, 2, 3]
    if layout == "NCDHW":
        dim_list = [0, 2, 3, 4]

    if output_mask[1]:
        scale_casted = __compute_scale(dim_list, dy_casted, x_casted, mean_casted, var_casted, eps, layout)
        diff_scale = ge.Cast(scale_casted, dst_type=meta_outputs[1].dtype)

    if output_mask[2]:
        offset_casted = ge.ReduceSum(dy_casted, axes=dim_list, keep_dims=False)
        diff_offset = ge.Cast(offset_casted, dst_type=meta_outputs[2].dtype)

    return diff_scale, diff_offset


def __mask_output(grad_in, diff_scale, diff_offset, output_mask):
    out_diff_scale = diff_scale if output_mask[1] else None
    out_diff_offset = diff_offset if output_mask[2] else None
    return grad_in, out_diff_scale, out_diff_offset


def __native_batch_norm_backward_3d(dim, input_size, eps, grad_out, invstd, mean, output_mask, train, weight, x):
    layout = "NCDHW"
    grad_out, x = __reshape_to_ncdhw(dim, input_size, grad_out, x)
    diff_scale, diff_offset = ge.BN3DTrainingUpdateGrad(grad_out, x, mean, invstd, epsilon=eps)
    specific_op_input_layout(diff_scale, indices=list(range(4)), layout=layout)
    specific_op_output_layout(diff_scale, indices=[0, 1], layout=layout)
    if train:
        grad_in = ge.BNTrainingReduceGrad(grad_out, x, diff_scale, diff_offset,
                                          weight, mean, invstd, epsilon=eps)
        specific_op_input_layout(grad_in, indices=list(range(7)), layout=layout)
        specific_op_output_layout(grad_in, indices=0, layout=layout)
    else:
        grad_in = ge.BNInferGrad(grads=grad_out, scale=weight, batch_variance=invstd, epsilon=eps)
        specific_op_input_layout(grad_in, indices=list(range(3)), layout=layout)
        specific_op_output_layout(grad_in, indices=0, layout=layout)
    grad_in = __ncdhw_to_origin(dim, input_size, grad_in)
    return grad_in, diff_scale, diff_offset


def __native_batch_norm_backward(dim, eps, grad_out, invstd, mean, output_mask, train, weight, x):
    layout = "NCHW"
    grad_out, x = __unsqueeze_inputs(dim, grad_out, x)
    diff_scale, diff_offset = ge.BNTrainingUpdateGrad(grad_out, x, mean, invstd, epsilon=eps)
    specific_op_input_layout(diff_scale, indices=list(range(4)), layout=layout)
    specific_op_output_layout(diff_scale, indices=[0, 1], layout=layout)
    if train:
        grad_in = ge.BNTrainingReduceGrad(grad_out, x, diff_scale, diff_offset,
                                          weight, mean, invstd, epsilon=eps)
        specific_op_input_layout(grad_in, indices=list(range(7)), layout=layout)
        specific_op_output_layout(grad_in, indices=0, layout=layout)
    else:
        grad_in = ge.BNInferGrad(grads=grad_out, scale=weight, batch_variance=invstd, epsilon=eps)
        specific_op_input_layout(grad_in, indices=list(range(3)), layout=layout)
        specific_op_output_layout(grad_in, indices=0, layout=layout)
    grad_in = __squeeze_grad_in(dim, grad_in)
    return grad_in, diff_scale, diff_offset


def __native_batch_norm_backward_regbase_3d(dim, input_size, grad_out, x, weight, running_mean, running_var,
                                            save_mean, save_invstd, train, eps, output_mask, meta_outputs):
    layout = "NCDHW"
    grad_out, x = __reshape_to_ncdhw(dim, input_size, grad_out, x)
    grad_in, diff_scale, diff_offset = ge.BatchNormGradV3(grad_out, x, weight, running_mean, running_var,
                                                          save_mean, save_invstd, is_training=train, epsilon=eps)
    specific_op_input_layout(grad_in, indices=[0, 1], layout=layout)
    specific_op_output_layout(grad_in, indices=0, layout=layout)
    if not train:
        diff_scale, diff_offset = \
            __calc_infer_output(grad_out, x, running_mean, running_var, eps, output_mask,
                                diff_scale, diff_offset, layout, meta_outputs)
    grad_in = __ncdhw_to_origin(dim, input_size, grad_in)
    return grad_in, diff_scale, diff_offset


def __native_batch_norm_backward_regbase(dim, grad_out, x, weight, running_mean, running_var,
                                         save_mean, save_invstd, train, eps, output_mask, meta_outputs):
    layout = "NCHW"
    grad_out, x = __unsqueeze_inputs(dim, grad_out, x)
    grad_in, diff_scale, diff_offset = ge.BatchNormGradV3(grad_out, x, weight, running_mean, running_var,
                                                          save_mean, save_invstd, is_training=train, epsilon=eps)
    specific_op_input_layout(grad_in, indices=[0, 1], layout=layout)
    specific_op_output_layout(grad_in, indices=0, layout=layout)
    if not train:
        diff_scale, diff_offset = \
            __calc_infer_output(grad_out, x, running_mean, running_var, eps, output_mask,
                                diff_scale, diff_offset, layout, meta_outputs)
    grad_in = __squeeze_grad_in(dim, grad_in)
    return grad_in, diff_scale, diff_offset


def __squeeze_grad_in(dim, grad_in):
    if dim == 2 or dim == 3:
        grad_in = ge.Squeeze(grad_in, axis=list(range(3, dim - 1, -1)))
    return grad_in


def __unsqueeze_inputs(dim, grad_out, x):
    if dim == 2 or dim == 3:
        x = ge.Unsqueeze(x, axes=list(range(dim, 4)))
        grad_out = ge.Unsqueeze(grad_out, axes=list(range(dim, 4)))
    return grad_out, x


def __reshape_to_ncdhw(dim, input_size, grad_out, x):
    if dim > 5:
        shape_list = ge.Pack([ge.Gather(input_size, 0), ge.Gather(input_size, 1), ge.Gather(input_size, 2),
                                ge.Gather(input_size, 3), -1], N=5, axis=0)
        grad_out = ge.Reshape(grad_out, shape_list)
        x = ge.Reshape(x, shape_list)
    return grad_out, x


def __ncdhw_to_origin(dim, input_size, grad_in):
    if dim > 5:
        grad_in = ge.Reshape(grad_in, input_size)
        specific_op_input_layout(grad_in, indices=[0, 1], layout="ND")
        specific_op_output_layout(grad_in, indices=0, layout="ND")
    return grad_in


@register_fx_node_ge_converter(torch.ops.aten.native_batch_norm_backward.out)
def conveter_aten_native_batch_norm_backward_out(
    grad_out: Tensor,
    input: Tensor,
    weight: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_invstd: Optional[Tensor],
    train: bool,
    eps: float,
    output_mask: List[bool],
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    out2: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::native_batch_norm_backward.out(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise RuntimeError("torch.ops.aten.native_batch_norm_backward.out ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")
