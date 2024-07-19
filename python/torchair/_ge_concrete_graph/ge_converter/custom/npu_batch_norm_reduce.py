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
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote, specific_op_input_layout, specific_op_output_layout


@declare_supported([
    Support(F32(32, 32, 1, 12544), 1e-7),
])
@register_fx_node_ge_converter(torch.ops.npu.batch_norm_reduce.default)
def conveter_npu_batch_norm_reduce_default(
    self: Tensor,
    eps: float,
    meta_outputs: TensorSpec = None
):  
    if self.rank != 4:
        raise NotImplementedError("torch.ops.npu.batch_norm_reduce.default ge_converter ", \
            "is only implemented for 4D input!")
    sum_output, square_sum = ge.BNTrainingReduce(self)
    specific_op_input_layout(sum_output, indices=0, layout="NCHW")
    specific_op_output_layout(sum_output, indices=[0, 1], layout="NCHW")
    return sum_output, square_sum
