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
        Support(F32(2, 2, 2, 2), F32(2), F32(2), F32(2), F32(2), training=True, momentum=0.1, eps=1e-5),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit_functional.default)
def conveter_aten__native_batch_norm_legit_functional_default(
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
    """NB: aten::_native_batch_norm_legit_functional(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor running_mean_out, Tensor running_var_out)"""
    if not training:
        raise NotImplementedError(
            "torch.ops.aten._native_batch_norm_legit_functional.default ge_converter is not implemented while training is False!"
        )
    sum_output, square_sum = ge.BNTrainingReduce(input)
    specific_op_input_layout(sum_output, indices=0, layout="NCHW")
    specific_op_output_layout(sum_output, indices=[0, 1], layout="NCHW")
    output, mean, variance, batch_mean, batch_variance = ge.BNTrainingUpdate(input, sum_output, \
        square_sum, weight, bias, running_mean, running_var, factor=momentum, epsilon=eps)
    specific_op_input_layout(output, indices=list(range(7)), layout="NCHW")
    specific_op_output_layout(output, indices=list(range(5)), layout="NCHW")
    return output, batch_mean, batch_variance, mean, variance
