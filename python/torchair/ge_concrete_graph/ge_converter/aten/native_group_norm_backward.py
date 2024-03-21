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
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import F32, Support
from torchair.ge_concrete_graph.utils import specific_op_input_layout, specific_op_output_layout
from torchair.ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F32(20, 6, 10, 10), F32(20, 6, 10, 10), F32(20, 3), F32(20, 3), weight=F32(6,), N=20, C=6, HxW=100,
                group=3, output_mask=[True, True, True]),
        Support(F32(20, 6, 10), F32(20, 6, 10), F32(20, 3), F32(20, 3), weight=F32(6, ), N=20, C=6, HxW=10,
                group=3, output_mask=[True, True, True])
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
        dy = ge.Reshape(grad_out, [N, C, HxW])
        x_reshape_1 = ge.Reshape(input, [N, C, HxW])
    else:
        dy = grad_out
        x_reshape_1 = input

    if weight is not None:
        dy_b = ge.Mul(dy, ge.Reshape(weight, [1, C, 1]))
    else:
        dy_b = dy

    x_reshaped = ge.Reshape(x_reshape_1, [1, N * group, -1 if N else 1])
    dy_reshaped = ge.Reshape(dy_b, [1, N * group, -1 if N else 1])
    mean_reshaped = ge.Reshape(mean, [N * group])

    rstd_square = ge.Mul(rstd, rstd)
    variances = ge.Sub(ge.Reciprocal(rstd_square), eps)
    rstd_reshaped = ge.Reshape(variances, [N * group])
    weight_opt = ge.Fill([N * group], dtype_promote(1, target_dtype=input.dtype))

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

    dx = ge.Reshape(grad_in, ge.Shape(x_reshape_1))
    dx = ge.Reshape(dx, ge.Shape(input))
    dgamma = dbeta = None

    if output_mask[1]:
        if mean.rank == 1:
            mean_broadcast = ge.Unsqueeze(ge.Reshape(mean, [N, group]), axes=[2])
            rstd_broadcast = ge.Unsqueeze(ge.Reshape(rstd, [N, group]), axes=[2])
        else:
            mean_broadcast = ge.Unsqueeze(mean, axes=[2])
            rstd_broadcast = ge.Unsqueeze(rstd, axes=[2])
        mean_broadcast = ge.BroadcastTo(mean_broadcast, ge.Pack([N, group, C / group], N=3))
        mean_broadcast = ge.Reshape(mean_broadcast, [N, C, 1])
        rstd_broadcast = ge.BroadcastTo(rstd_broadcast, ge.Pack([N, group, C / group], N=3))
        rstd_broadcast = ge.Reshape(rstd_broadcast, [N, C, 1])
        x_hat = ge.Mul(ge.Sub(x_reshape_1, mean_broadcast), rstd_broadcast)
        dgamma = ge.Mul(dy, x_hat)
        dgamma = ge.ReduceSum(dgamma, axes=[0, 2], keep_dims=False)
    if output_mask[2]:
        dbeta = ge.ReduceSum(dy, axes=[0, 2], keep_dims=False)
    return dx, dgamma, dbeta


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
    raise NotImplementedError("torch.ops.aten.native_group_norm_backward.out ge_converter is not implemented!")
