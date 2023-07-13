import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
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


@register_fx_node_ge_converter(torch.ops.aten.nan_to_num_.default)
def conveter_aten_nan_to_num__default(
        self: Tensor,
        nan: Optional[float] = None,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.nan_to_num_.default ge converter is not implement!")


