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


@register_fx_node_ge_converter(torch.ops.aten.log_sigmoid_forward.default)
def conveter_aten_log_sigmoid_forward_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)"""
    raise NotImplementedError("torch.ops.aten.log_sigmoid_forward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log_sigmoid_forward.output)
def conveter_aten_log_sigmoid_forward_output(
    self: Tensor,
    *,
    output: Tensor = None,
    buffer: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::log_sigmoid_forward.output(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))"""
    raise NotImplementedError("torch.ops.aten.log_sigmoid_forward.output ge_converter is not implemented!")
