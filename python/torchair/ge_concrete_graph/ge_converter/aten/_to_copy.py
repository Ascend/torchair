import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, torch_type_to_ge_type
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


@register_fx_node_ge_converter(torch.ops.aten._to_copy.default)
def conveter_aten__to_copy_default(
        self: Tensor,
        *,
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        non_blocking: bool = False,
        memory_format: Optional[int] = None,
        meta_outputs: Any = None):
    """ NB: aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor """
    if layout is not None or device is not None or pin_memory is not None \
            or non_blocking is not None or memory_format is not None:
        # TODO: fix this case
        print(f"[warning] torch.ops.aten._to_copy.default have some unprocessed parameters or cases!")

    if dtype is None:
        return ge.Identity(self)
    else:
        return ge.Cast(self, dst_type=torch_type_to_ge_type(dtype))


@register_fx_node_ge_converter(torch.ops.aten._to_copy.out)
def conveter_aten__to_copy_out(
        self: Tensor,
        *,
        non_blocking: bool = False,
        memory_format: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::_to_copy.out(Tensor self, *, bool non_blocking=False, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten._to_copy.out ge converter is not implement!")


