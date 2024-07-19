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
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F16(8192, 320), F16(320, 2560), F16(2560, 320), activation="relu", inner_precise=1, output_dtype=None),
    Support(F16(8192, 320), F16(320, 2560), F16(2560, 320), activation="gelu", inner_precise=1, output_dtype=None),
    Support(F16(8192, 320), F16(320, 2560), F16(2560, 320), activation="fastgelu", inner_precise=1, output_dtype=None),
    Support(F16(8192, 320), F16(320, 2560), F16(2560, 320), activation="silu", inner_precise=1, output_dtype=None),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_ffn.default)
def conveter_npu_npu_ffn(
    x: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    activation: str,
    *,
    expert_tokens: Optional[List[int]] = None,
    expert_tokens_index: Optional[List[int]] = None,
    bias1: Optional[Tensor] = None,
    bias2: Optional[Tensor] = None,
    scale: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
    deq_scale1: Optional[Tensor] = None,
    deq_scale2: Optional[Tensor] = None,
    antiquant_scale1: Optional[Tensor] = None,
    antiquant_scale2: Optional[Tensor] = None,
    antiquant_offset1: Optional[Tensor] = None,
    antiquant_offset2: Optional[Tensor] = None,
    inner_precise: Optional[int] = 0,
    output_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_ffn(Tensor x, Tensor weight1, Tensor weight2, str activation, *, int[]? expert_tokens=None,
                        int[]? expert_tokens_index=None, Tensor? bias1=None, Tensor? bias2=None, Tensor? scale=None,
                        Tensor? offset=None, Tensor? deq_scale1=None, Tensor? deq_scale2=None,
                        Tensor? antiquant_scale1=None, Tensor? antiquant_scale2=None, Tensor? antiquant_offset1=None,
                        Tensor? antiquant_offset2=None, int? inner_precise=None, ScalarType? output_dtype=None)
                        -> Tensor
    """
    tokens_index_flag = False
    if expert_tokens is not None and expert_tokens_index is not None:
        raise ValueError("torch.ops.npu.npu_ffn.default: " \
                         "Cannot assign the value to expert_tokens and expert_tokens_index simultaneously!")
    elif expert_tokens_index is not None:
        tokens_index_flag = True
        expert_tokens = expert_tokens_index

    y_dtype = -1
    if x.dtype == DataType.DT_INT8 and output_dtype is not None:
        if output_dtype == torch.float16:
            y_dtype = 0
        elif output_dtype == torch.bfloat16:
            y_dtype = 1
        else:
            raise NotImplementedError("torch.ops.npu.npu_ffn.default: In the quant scenario, " \
                                      "output_dtype should be float16 or bfloat16, otherwise it should be None!")

    if expert_tokens is not None:
        expert_tokens = dtype_promote(expert_tokens, target_dtype=torch.int64)
    return ge.FFN(x, weight1, weight2, expert_tokens=expert_tokens, bias1=bias1, bias2=bias2, scale=scale,
                  offset=offset, deq_scale1=deq_scale1, deq_scale2=deq_scale2, antiquant_scale1=antiquant_scale1,
                  antiquant_scale2=antiquant_scale2, antiquant_offset1=antiquant_offset1,
                  antiquant_offset2=antiquant_offset2, activation=activation, inner_precise=inner_precise,
                  output_dtype=y_dtype, tokens_index_flag=tokens_index_flag)