from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import torch
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, strided
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported
from torchair._ge_concrete_graph.supported_declaration import Support, F32, F16


@declare_supported([
    Support(F32(3), F32(3)),
    Support(F32(3), F16(3)),
    Support(F32(5, 3, 4, 1), F32(3, 1, 1)),
    Support(F32(5, 3, 4, 1), F32(3, 1, 1), non_blocking=False),
])
@register_fx_node_ge_converter(torch.ops.aten.copy.default)
def conveter_aten_copy_default(
        self: Tensor,
        src: Tensor,
        non_blocking: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor """
    # GE graph are not allowed to be copied across devices (if they do exist, the graph should be broken),
    # so aten's copy in GE graph is semantically degraded to a copy of values on the same device.

    if self is src:
        return ge.Identity(src)
    else:
        src = dtype_promote(src, target_dtype=self.dtype)

        if str(self._symsize) == str(src._symsize):
            # The input shapes and output shapes are consistent and no need to broadcast
            return src

        return ge.BroadcastTo(src, ge.Shape(self))


@register_fx_node_ge_converter(torch.ops.aten.copy.out)
def conveter_aten_copy_out(
        self: Tensor,
        src: Tensor,
        non_blocking: bool = False,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::copy(Tensor self, Tensor src, bool non_blocking=False, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError(
        "torch.ops.aten.copy.out ge converter is not implement!")
