from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, \
    U8, BOOL, Support


@register_fx_node_ge_converter(torch.ops.npu.npu_batch_gather_matmul.default)
def conveter_npu_batch_gather_matmul_default(
    y: Tensor,
    x: Tensor,
    weight_b: Tensor,
    indices: Tensor,
    weight_a: Optional[Tensor] = None,
    layer_idx: int = 0,
    scale: float = 1e-3,
    y_offset: int = 0,
    y_slice_size: int = -1,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_batch_gather_matmul(Tensor y, Tensor x, Tensor weight_b, Tensor indices, Tensor? weight_a=None,
                                    int layer_idx=0, float scale=1e-3, int y_offset=0, int y_slice_size=-1) -> Tensor"""
    return ge.AddLora(y, x, weightA=weight_a, weightB=weight_b, indices=indices,
                      layer_idx=layer_idx, scale=scale, y_offset=y_offset, y_slice_size=y_slice_size)
