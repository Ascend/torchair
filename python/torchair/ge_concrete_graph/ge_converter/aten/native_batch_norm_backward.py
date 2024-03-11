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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
from torchair.ge_concrete_graph.utils import specific_op_input_layout, \
    specific_op_output_layout
from torchair.ge_concrete_graph.supported_declaration import F32, F16, Support


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
    if train:
        mean = save_mean
        invstd = save_invstd
    else:
        mean = running_mean
        invstd = running_var
    if not all(output_mask):
        raise RuntimeError("torch.ops.aten.native_batch_norm_backward.default "
                           "ge_converter is not implemented while output_mask are not all True!")
    dim = input.rank
    if dim > 5 or dim < 2:
        raise RuntimeError("torch.ops.aten.native_batch_norm_backward.default "
                           "ge_converter is not implemented while input dim > 5 or input dim <2!")
    input_size = ge.Shape(input, dtype=DataType.DT_INT32)
    if weight is None:
        weight = ge.Fill(ge.Gather(input_size, ge.Cast(2 if dim == 5 else 1, dst_type=DataType.DT_INT32)),
                         ge.Cast(1., dst_type=input.dtype))
    if dim <= 4:
        if dim == 2 or dim == 3:
            input = ge.Unsqueeze(input, axes=list(range(dim, 4)))
            grad_out = ge.Unsqueeze(grad_out, axes=list(range(dim, 4)))

        diff_scale, diff_offset = ge.BNTrainingUpdateGrad(grad_out, input, mean, invstd, epsilon=eps)
        specific_op_input_layout(diff_scale, indices=list(range(4)), layout="NCHW")
        specific_op_output_layout(diff_scale, indices=[0, 1], layout="NCHW")

        if train:
            grad_in = ge.BNTrainingReduceGrad(grad_out, input, diff_scale, diff_offset,
                                              weight, mean, invstd, epsilon=eps)
            specific_op_input_layout(grad_in, indices=list(range(7)), layout="NCHW")
            specific_op_output_layout(grad_in, indices=0, layout="NCHW")
            if dim == 2 or dim == 3:
                grad_in = ge.Squeeze(grad_in, axis=list(range(3, dim - 1, -1)))
            return grad_in, diff_scale, diff_offset
    else:
        diff_scale, diff_offset = ge.BN3DTrainingUpdateGrad(grad_out, input, mean, invstd, epsilon=eps)
        if train:
            specific_op_input_layout(diff_scale, indices=list(range(4)), layout="NCDHW")
            specific_op_output_layout(diff_scale, indices=[0, 1], layout="NCDHW")
            grad_in = ge.BNTrainingReduceGrad(grad_out, input, diff_scale, diff_offset,
                                            weight, mean, invstd, epsilon=eps)
            specific_op_input_layout(grad_in, indices=list(range(7)), layout="NCDHW")
            specific_op_output_layout(grad_in, indices=0, layout="NCDHW")
            return grad_in, diff_scale, diff_offset

    grad_in = ge.BNInferGrad(grads=grad_out, scale=weight, batch_variance=invstd, epsilon=eps)
    return grad_in, diff_scale, diff_offset


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
