import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.nvprims.var_mean.default)
def conveter_nvprims_var_mean_default(
        inp: Tensor,
        unbiased: bool,
        meta_outputs: Any = None):
    """ NB: nvprims::var_mean(Tensor inp, bool unbiased) -> (Tensor, Tensor) """
    raise NotImplementedError("torch.ops.nvprims.var_mean.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.nvprims.var_mean.main)
def conveter_nvprims_var_mean_main(
        inp: Tensor,
        dim: Optional[List[int]] = None,
        unbiased: Optional[bool] = None,
        keepdim: bool = False,
        *,
        correction: Optional[float] = None,
        meta_outputs: Any = None):
    """ NB: nvprims::var_mean.main(Tensor inp, int[1]? dim=None, bool? unbiased=None, bool keepdim=False, *, float? correction=None) -> (Tensor, Tensor) """
    raise NotImplementedError("torch.ops.nvprims.var_mean.main ge converter is not implement!")


