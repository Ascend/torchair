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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.rrelu.default)
def conveter_aten_rrelu_default(
    self: Tensor,
    lower: Union[Number, Tensor] = 0.125,
    upper: Union[Number, Tensor] = 0.3333333333333333,
    training: bool = False,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.rrelu.default ge_converter is not implemented!")
