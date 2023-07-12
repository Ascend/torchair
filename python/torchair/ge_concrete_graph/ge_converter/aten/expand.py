import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.fx2ge_converter import register_testcase
from torchair.ge_concrete_graph.testing_utils import *
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.utils import dtype_promote
from torch import contiguous_format, Generator, inf, memory_format, strided
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


@register_testcase([
    TestInput(F32(3, 1), size=[3, 4]),
    TestInput(F16(3, 1, 2), size=[3, 5, 2]),
])
@register_fx_node_ge_converter(torch.ops.aten.expand.default)
def conveter_aten_expand_default(
        self: Tensor,
        size: Union[List[int], Tensor],
        *,
        implicit: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a) """
    if implicit:
        raise NotImplementedError("torch.ops.aten.expand.default ge converter is not implement, "
                                  "when param implicit is True.")

    if isinstance(size, Tensor):
        return ge.BroadcastTo(self, size)
    else:
        positive_size = []
        for i in range(len(size)):
            if size[i] > 0:
                positive_size.append(size[i])
            else:
                positive_size.append(size[i] + meta_outputs.size[i])
        return ge.BroadcastTo(self, positive_size)
