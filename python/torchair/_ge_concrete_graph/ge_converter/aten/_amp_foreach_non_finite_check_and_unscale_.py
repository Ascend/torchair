from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._amp_foreach_non_finite_check_and_unscale_.default)
def conveter_aten__amp_foreach_non_finite_check_and_unscale__default(
    self: List[Tensor], found_inf: Tensor, inv_scale: Tensor):
    """NB: aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self, Tensor(b!) found_inf, Tensor inv_scale) -> ()"""
    raise NotImplementedError(
        "torch.ops.aten._amp_foreach_non_finite_check_and_unscale_.default ge_converter is not implemented!"
    )
