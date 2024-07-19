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


@register_fx_node_ge_converter(torch.ops.aten.hinge_embedding_loss.default)
def conveter_aten_hinge_embedding_loss_default(
    self: Tensor,
    target: Tensor,
    margin: float = 1.0,
    reduction: int = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1., int reduction=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.hinge_embedding_loss.default ge_converter is not implemented!")
