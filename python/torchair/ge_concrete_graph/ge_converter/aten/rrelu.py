
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
@register_fx_node_ge_converter(torch.ops.aten.rrelu.default)
def conveter_aten_rrelu_default(
        self: Tensor,
        lower: Union[Number, Tensor] = 0.125,
        upper: Union[Number, Tensor] = 0.3333333333333333,
        training: bool = False,
        generator: Optional[Generator] = None,
        meta_outputs: Any = None):
    """ NB: aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.rrelu.default ge converter is not implement!")


