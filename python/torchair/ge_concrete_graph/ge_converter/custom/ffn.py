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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F16(8192, 320), F16(320, 2560), F16(2560, 320), activation="relu", inner_precise=1),
        Support(F16(8192, 320), F16(320, 2560), F16(2560, 320), activation="gelu", inner_precise=1),
        Support(F16(8192, 320), F16(320, 2560), F16(2560, 320), activation="fastgelu", inner_precise=1),
        Support(F16(8192, 320), F16(320, 2560), F16(2560, 320), activation="silu", inner_precise=1),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_ffn.default)
def conveter_npu_npu_ffn(
    x: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    activation: str,
    expert_tokens: Optional[Union[List[int], Tensor]] = None,
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
    inner_precise: Optional[int] = 0
):
    """NB: npu::npu_ffn(Tensor x, Tensor weight1, Tensor weight2, str activation, int[]? expert_tokens=None,
                        Tensor? bias1=None, Tensor? bias2=None, Tensor? scale=None, Tensor? offset=None,
                        Tensor? deq_scale1=None, Tensor? deq_scale2=None, Tensor? antiquant_scale1=None,
                        Tensor? antiquant_scale2=None, Tensor? antiquant_offset1=None,
                        Tensor? antiquant_offset2=None, int? inner_precise=None) -> Tensor
    """
    if expert_tokens is not None and isinstance(expert_tokens, Tensor):
        raise NotImplementedError("FFN is not implemented while expert_tokens is Tensor!")
    if scale is not None or offset is not None or deq_scale1 is not None or deq_scale2 is not None or \
        antiquant_scale1 is not None or antiquant_scale2 is not None or antiquant_offset1 is not None or \
        antiquant_offset2 is not None:
        raise NotImplementedError("FFN dose not support quant case and antiquant case now!")

    return ge.FFN(x, weight1, weight2, expert_tokens=expert_tokens, bias1=bias1, bias2=bias2, scale=None,
        offset=None, deq_scale1=None, deq_scale2=None, antiquant_scale1=None, antiquant_scale2=None,
        antiquant_offset1=None, antiquant_offset2=None, activation=activation, inner_precise=inner_precise)
