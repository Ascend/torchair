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
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec, is_sym
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(1024, 1024), (2, 2), (256, 256), 0),
    Support(F32(1024, 1024), (2, 2), (256, 256), 1),
    Support(F16(1024, 1024), (2, 2), (256, 256), 0),
    Support(F16(1024, 1024), (2, 2), (256, 256), 1),
    Support(F32(3, 3), (2, 2), (1, 2)),
    Support(F32(3, 3), (2, 2), (1, 2), 1),
    Support(F32(3, 3), (2, 2), (1, 2), 0),
    Support(F16(8, 2, 512, 1), [8, 3, 512, 1], [1024, 256, 1, 1]),
    Support(F16(96, 2, 512, 64), [96, 3, 512, 64], [64, 1572864, 6144, 1]),
])
@register_fx_node_ge_converter(torch.ops.aten.as_strided.default)
def conveter_aten_as_strided_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    stride: Union[List[int], Tensor],
    storage_offset: Optional[Union[int, Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)"""

    storage_offset = 0 if storage_offset is None else storage_offset
    # When the last dimension of the input tensor stride is greater than 256, we use
    # AsStrided + Transpose instead of AsStrided to get better performance.
    # Stride will be List[int] in static mode, be Tensor in dynamic mode.
    if isinstance(stride, List) and isinstance(size, List) and stride[-1] >= 256:
        tensor_dim = len(stride)
        
        stride_perm_shape_set = set()

        for i in range(tensor_dim):
            stride_perm_shape_set.add((stride[i], (i, size[i])))
        
        output_stride = []
        output_shape = []
        output_perm_origin = []

        for item in sorted(stride_perm_shape_set, reverse=True):
            output_stride.append(item[0])
            output_shape.append(item[1][1])
            output_perm_origin.append(item[1][0])

        storage_offset = dtype_promote(storage_offset, target_dtype=DataType.DT_INT64)
        result_out = ge.AsStrided(self, output_shape, output_stride, storage_offset=storage_offset)
        output_perm = [0] * tensor_dim
        for i in range(tensor_dim):
            output_perm[output_perm_origin[i]] = i

        return ge.Transpose(result_out, output_perm)
    else:
        return ge.AsStrided(self, size, stride, storage_offset=storage_offset)
