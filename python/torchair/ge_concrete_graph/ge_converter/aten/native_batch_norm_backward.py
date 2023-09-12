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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.utils import specific_op_input_layout, \
    specific_op_output_layout
from torchair.ge_concrete_graph.supported_declaration import F32, F16, Support


@declare_supported(
    [
        Support(F32(2, 2, 2, 2), F32(2, 2, 2, 2), F32(2), F32(2), F32(2), F32(2), F32(2), train=True, eps=1e-5, output_mask=[True, True, True]),
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
    if not train:
        """ running_mean and running_var is only used when train is False.""" 
        raise NotImplementedError(
            "torch.ops.aten._native_batch_norm_legit_functional.default ge_converter is not implemented while train is False!"
        )
    if save_mean is None or save_invstd is None:
        raise NotImplementedError(
            "torch.ops.aten._native_batch_norm_legit_functional.default ge_converter is not implemented while save_mean is None or save_invstd is None!"
        )
    if not all(output_mask):
        raise NotImplementedError(
            "torch.ops.aten._native_batch_norm_legit_functional.default ge_converter is not implemented while output_mask are not all True!"
        )
    diff_scale, diff_offset = ge.BNTrainingUpdateGrad(grad_out, input, save_mean, save_invstd, epsilon=eps)
    specific_op_input_layout(diff_scale, indices=list(range(4)), layout="NCHW")
    specific_op_output_layout(diff_scale, indices=[0, 1], layout="NCHW")
    grad_in = ge.BNTrainingReduceGrad(grad_out, input, diff_scale, diff_offset, weight, save_mean, save_invstd, epsilon=eps)
    specific_op_input_layout(grad_in, indices=list(range(7)), layout="NCHW")
    specific_op_output_layout(grad_in, indices=0, layout="NCHW")
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
    raise NotImplementedError("torch.ops.aten.native_batch_norm_backward.out ge_converter is not implemented!")
