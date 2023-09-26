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
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


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
        raise NotImplementedError(
            "torch.ops.aten.expand.default ge_converter is not implemented, " "when param implicit is True."
        )

    if isinstance(size, Tensor):
        return ge.BroadcastTo(self, size)
    else:
        positive_size = []
        for i in range(len(size)):
            if size[i] == -1:
                positive_size.append(meta_outputs.size[i])
            else:
                positive_size.append(size[i])
                
        if str(self._symsize) != str(positive_size):
            return ge.BroadcastTo(self, positive_size)
        return self
