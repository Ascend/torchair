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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


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
    raise NotImplementedError(
        "torch.ops.aten._native_batch_norm_legit_functional.default ge_converter is not implemented!"
    )
