from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(1, 3, 3), [1, 1]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.replication_pad1d.default)
def conveter_aten_replication_pad1d_default(
        self: Tensor, padding: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::replication_pad1d(Tensor self, SymInt[4] padding) -> Tensor"""
    if isinstance(padding, Tensor):
        raise NotImplementedError(
            "When padding is Tensor, torch.ops.aten.replication_pad1d.default ge_converter is not implemented!")
    if len(padding) < 2:
        raise AssertionError("padding length shoud be at least 2")
    paddings = [0, 0, 0, 0, 0, 0, padding[0], padding[1]]
    self_cp = ge.Unsqueeze(self, axes=[0])
    result = ge.PadV3(self_cp, paddings, [0], mode="edge", paddings_contiguous=True)
    return ge.Squeeze(result, axis=[0])


@register_fx_node_ge_converter(torch.ops.aten.replication_pad1d.out)
def conveter_aten_replication_pad1d_out(
        self: Tensor,
        padding: Union[List[int], Tensor],
        *,
        out: Tensor = None,
        meta_outputs: TensorSpec = None
):
    """NB: aten::replication_pad1d.out(Tensor self, SymInt[4] padding, *, Tensor(a!) out) -> Tensor(a!)"""
    raise AssertionError(
        "torch.ops.aten.replication_pad1d.out is redundant before pytorch 2.1.0, might be supported in future version.")
