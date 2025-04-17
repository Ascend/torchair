from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import F32, Support
from torchair._ge_concrete_graph.utils import specific_op_input_layout, specific_op_output_layout
from torchair._ge_concrete_graph.utils import dtype_promote


def _get_numel(inputs):
    numel = 1
    for symsize in inputs.symsize:
        numel *= symsize
    return numel


def _output_mask(inputs, mean, rstd, n, c, group, output_mask, grad_in, x_reshape_1, dy):
    dx = ge.Reshape(grad_in, ge.Shape(inputs))

    zero_contain = 0
    if inputs.symsize is not None and _get_numel(inputs) == 0:
        zero_contain = 1

    if zero_contain == 1:
        return dx, ge.Fill([c], ge.Cast(0., dst_type=mean.dtype)) if output_mask[1] else None, \
                ge.Fill([c], ge.Cast(0., dst_type=rstd.dtype)) if output_mask[2] else None

    dgamma = dbeta = None

    if output_mask[1]:
        if mean.rank == 1:
            pack_1 = ge.Pack([n, group], N=2, axis=0)
            mean_broadcast = ge.Unsqueeze(ge.Reshape(mean, pack_1), axes=[2])
            rstd_broadcast = ge.Unsqueeze(ge.Reshape(rstd, pack_1), axes=[2])
        else:
            mean_broadcast = ge.Unsqueeze(mean, axes=[2])
            rstd_broadcast = ge.Unsqueeze(rstd, axes=[2])

        pack_2 = ge.Pack([n, group, ge.RealDiv(c, group)], N=3)
        pack_3 = ge.Pack([n, c, 1], N=3, axis=0)
        mean_broadcast = ge.BroadcastTo(mean_broadcast, pack_2)
        mean_broadcast = ge.Reshape(mean_broadcast, pack_3)
        rstd_broadcast = ge.BroadcastTo(rstd_broadcast, pack_2)
        rstd_broadcast = ge.Reshape(rstd_broadcast, pack_3)
        x_hat = ge.Mul(ge.Sub(x_reshape_1, mean_broadcast), rstd_broadcast)
        dgamma = ge.Mul(dy, x_hat)
        dgamma = ge.ReduceSum(dgamma, axes=[0, 2], keep_dims=False)
    if output_mask[2]:
        dbeta = ge.ReduceSum(dy, axes=[0, 2], keep_dims=False)

    return dx, dgamma, dbeta


@declare_supported(
    [
        Support(F32(20, 6, 10, 10), F32(20, 6, 10, 10), F32(20, 3), F32(20, 3), weight=F32(6, ), N=20, C=6, HxW=100,
                group=3, output_mask=[True, True, True]),
        Support(F32(20, 6, 10), F32(20, 6, 10), F32(20, 3), F32(20, 3), weight=F32(6, ), N=20, C=6, HxW=10,
                group=3, output_mask=[True, True, True]),
        Support(F32(0, 6, 10), F32(0, 6, 10), F32(0, 3), F32(0, 3), weight=F32(6, ), N=0, C=6, HxW=10,
                group=3, output_mask=[True, True, True]),
        Support(F32(9, 1), F32(9, 1), F32(9, 1), F32(9, 1), weight=F32(1, ), N=9, C=1, HxW=1,
                group=1, output_mask=[True, False, False]),
        Support(F32(0, 10), F32(0, 10), F32(0, 1), F32(0, 1), weight=F32(10, ), N=0, C=10, HxW=1,
                group=1, output_mask=[True, False, False]),
    ]
)
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
    eps = 1e-5
    dim = input.rank
    if dim != 3:
        shape_list_0 = ge.Pack([N, C, HxW], N=3, axis=0)
        dy = ge.Reshape(grad_out, shape_list_0)
        x_reshape_1 = ge.Reshape(input, shape_list_0)
    else:
        dy = grad_out
        x_reshape_1 = input

    if weight is not None:
        dy_b = ge.Mul(dy, ge.Reshape(weight, ge.Pack([1, C, 1], N=3, axis=0)))
    else:
        dy_b = dy

    group_n = ge.Mul(N, group)
    shape_list_1 = ge.Pack([1, group_n, -1 if N else 1], N=3, axis=0)
    shape_list_2 = ge.Pack([group_n], N=1, axis=0)
    x_reshaped = ge.Reshape(x_reshape_1, shape_list_1)
    dy_reshaped = ge.Reshape(dy_b, shape_list_1)
    mean_reshaped = ge.Reshape(mean, shape_list_2)

    rstd_square = ge.Mul(rstd, rstd)
    variances = ge.Sub(ge.Reciprocal(rstd_square), eps)
    rstd_reshaped = ge.Reshape(variances, shape_list_2)
    weight_opt = ge.Fill(shape_list_2, dtype_promote(1, target_dtype=input.dtype))

    x_reshaped = ge.Unsqueeze(x_reshaped, axes=list(range(3, 4)))
    dy_reshaped = ge.Unsqueeze(dy_reshaped, axes=list(range(3, 4)))

    diff_scale, diff_offset = ge.BNTrainingUpdateGrad(dy_reshaped, x_reshaped,
                                                      mean_reshaped, rstd_reshaped, epsilon=eps)
    specific_op_input_layout(diff_scale, indices=list(range(4)), layout="NCHW")
    specific_op_output_layout(diff_scale, indices=[0, 1], layout="NCHW")
    grad_in = ge.BNTrainingReduceGrad(dy_reshaped, x_reshaped, diff_scale, diff_offset,
                                      weight_opt, mean_reshaped, rstd_reshaped, epsilon=eps)
    specific_op_input_layout(grad_in, indices=list(range(7)), layout="NCHW")
    specific_op_output_layout(grad_in, indices=0, layout="NCHW")
    grad_in = ge.Squeeze(grad_in, axis=list(range(3, 2, -1)))

    return _output_mask(input, mean, rstd, N, C, group, output_mask, grad_in, x_reshape_1, dy)


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
    raise RuntimeError("torch.ops.aten.native_group_norm_backward.out is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")
