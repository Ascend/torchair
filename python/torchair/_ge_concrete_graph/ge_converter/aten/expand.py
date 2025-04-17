from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F32(3, 1), size=[3, 4]),
        Support(F16(3, 1, 2), size=[-1, 5, -1]),
        Support(F16(2, 3), size=[2, 2, 3]),
        Support(F16(3, 4, 2), size=[4, -1, -1, -1])
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.expand.default)
def conveter_aten_expand_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    *,
    implicit: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)"""
    if implicit:
        raise RuntimeError(
            "torch.ops.aten.expand.default ge_converter is not implemented when param implicit is True."
        )
    # performance optimization: if the input and output symbolic shape is equal, do not broadcast
    if hasattr(self, "_symsize") and meta_outputs is not None and hasattr(meta_outputs, "_symsize"):
        if str(self._symsize) == str(meta_outputs._symsize):
            return self
    return ge.BroadcastTo(self, size)

