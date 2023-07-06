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


@register_fx_node_ge_converter(torch.ops.aten._amp_foreach_non_finite_check_and_unscale_.default)
def conveter_aten__amp_foreach_non_finite_check_and_unscale__default(
        self: List[Tensor],
        found_inf: Tensor,
        inv_scale: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self, Tensor(b!) found_inf, Tensor inv_scale) -> () """
    raise NotImplementedError("torch.ops.aten._amp_foreach_non_finite_check_and_unscale_.default ge converter is not implement!")


