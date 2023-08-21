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


@register_fx_node_ge_converter(torch.ops.nvprims.var_mean.default)
def conveter_nvprims_var_mean_default(
    inp: Tensor, unbiased: bool, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: nvprims::var_mean(Tensor inp, bool unbiased) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.nvprims.var_mean.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.nvprims.var_mean.main)
def conveter_nvprims_var_mean_main(
    inp: Tensor,
    dim: Optional[List[int]] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[float] = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: nvprims::var_mean.main(Tensor inp, int[1]? dim=None, bool? unbiased=None, bool keepdim=False, *, float? correction=None) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.nvprims.var_mean.main ge_converter is not implemented!")
