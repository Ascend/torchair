from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, \
    I64, I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote    


@declare_supported(
    [
        Support(F32(4, 2)),
        Support(F32(4, 3, 2))
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.view_as_complex.default)
def conveter_aten_view_as_complex_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::view_as_complex(Tensor(a) self) -> Tensor(a)"""
    self = dtype_promote(self, target_dtype=DataType.DT_FLOAT) 
    real = ge.GatherV2(self, 0, [-1], negative_index_support=True)
    imag = ge.GatherV2(self, 1, [-1], negative_index_support=True)
    return ge.Complex(real, imag)
