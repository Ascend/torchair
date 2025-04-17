from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, torch_type_to_ge_type, \
    declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote, specific_op_input_layout, specific_op_output_layout


@declare_supported([
    Support(F32(32, 32, 112, 112), F32(32), F32(32), F32(32), F32(32), 1e-7),
])
@register_fx_node_ge_converter(torch.ops.aten.batch_norm_elemt.default)
def conveter_aten_batch_norm_elemt_default(
    inp: Tensor, 
    weight: Optional[Tensor], 
    bias: Optional[Tensor], 
    running_mean: Tensor, 
    running_var: Tensor, 
    eps: float, 
    meta_outputs: TensorSpec = None
):
    if inp.rank != 4:
        raise NotImplementedError("torch.ops.aten.batch_norm_elemt.default ", \
            "ge_converter is only implemented for 4D input!")
    var = ge.Mul(running_var, running_var)
    var = ge.Div(1.0, var)
    var = ge.Sub(var, eps)

    output = ge.BNInfer(inp, weight, bias, running_mean, var, epsilon=eps)
    specific_op_input_layout(output, indices=list(range(5)), layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    return output
