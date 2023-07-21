import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported
from torchair.ge_concrete_graph.supported_declaration import *
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


@declare_supported([
    Support(F32(2, 2, 2), F32(2, 2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.bmm.default)
def conveter_aten_bmm_default(
        self: Tensor,
        mat2: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bmm(Tensor self, Tensor mat2) -> Tensor """
    self, mat2 = dtype_promote(self, mat2, target_dtype = meta_outputs.dtype)
    return ge.BatchMatMul(self, mat2)


@register_fx_node_ge_converter(torch.ops.aten.bmm.out)
def conveter_aten_bmm_out(
        self: Tensor,
        mat2: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bmm.out ge converter is not implement!")


