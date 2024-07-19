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

import sys
import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.supported_declaration import F32, F16, I32, U8, Support


@declare_supported(
    [
        Support([F16(4, 4096, 256), 
                 F16(4, 4096, 256), 
                 F16(4, 4096, 256), 
                 F16(4, 4096, 256), 
                 F16(4, 4096, 256), 
                 F16(4, 4096, 256), 
                 F16(4, 4096, 256), 
                 F16(4, 4096, 256)], I32(8), F16(8, 4, 1, 256), U8(8),
            reduce="update", axis=-2),
        Support([F32(4, 4096, 256), 
                 F32(4, 4096, 256), 
                 F32(4, 4096, 256), 
                 F32(4, 4096, 256), 
                 F32(4, 4096, 256), 
                 F32(4, 4096, 256), 
                 F32(4, 4096, 256), 
                 F32(4, 4096, 256)], I32(8), F32(8, 4, 1, 256), U8(8),
            reduce="update", axis=-2),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_scatter_list.default)
def conveter_npu_npu_scatter_list_default(var: List[Tensor], indice: Tensor, updates: Tensor,
    mask: Optional[Tensor], reduce: str = 'update', axis: int = -2, meta_outputs: TensorSpec = None):
    """NB: func: npu_scatter_list(Tensor data[], Tensor indices, 
        Tensor updates, Tensor mask, int axis, string update) -> Tensor
    Warning: kernel [scatter_list] is a out-of-place op, This current usage may cause the input to be changed
    unexpectedly, and the caller needs to pay attention to this feature.
    """
    var_temp = []
    for item in var:
        var_temp.append(ge.TensorMove(item))

    return ge.ScatterList(var_temp, indice, updates, mask, reduce=reduce, axis=axis)
