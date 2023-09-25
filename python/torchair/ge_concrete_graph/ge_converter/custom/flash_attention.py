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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        # 支持输入q、k、v，BSH三维格式,q:b=1, s=2048, h=40*128;k/v:b=1, s=2048, h=40*128;
        Support(F16(1, 2048, 40 * 128), F16(1, 2048, 40 * 128), F16(1, 2048, 40 * 128),
            num_heads=40, input_layout="BSH"),
        # 支持输入q、k、v，BNSD四维格式
        Support(F16(1, 40, 2048, 128), F16(1, 40, 2048, 128), F16(1, 40, 2048, 128),
            num_heads=40, input_layout="BNSD"),
        # 支持设置scale_value
        Support(F16(1, 40, 2048, 128), F16(1, 40, 2048, 128), F16(1, 40, 2048, 128),
            input_layout="BNSD", num_heads=40, scale_value=0.0884),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_prompt_flash_attention.default)
def convert_npu_npu_prompt_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    padding_mask: Optional[Tensor] = None,
    atten_mask: Optional[Tensor] = None,
    actual_seq_lengths: Optional[Union[List[int], Tensor]] = None,
    num_heads: int = 1,
    scale_value: float = 1.0,
    pre_tokens: int = 2147473647,
    next_tokens: int = 0,
    input_layout: str = "BSH",
    num_key_value_heads: int = 0,
    meta_outputs: TensorSpec = None,
):

    '''NB: npu::npu_prompt_flash_attention(Tensor query, Tensor key, Tensor value, *, Tensor? padding_mask=None, Tensor? atten_mask=None, int[]? actual_seq_lengths=None, int num_heads=1, float scale_value=1.0, int pre_tokens=2147473647, int next_tokens=0, str input_layout="BSH", int num_key_value_heads=0) -> Tensor'''
    if actual_seq_lengths is not None and isinstance(actual_seq_lengths, Tensor):
        raise NotImplementedError("PromptFlashAttention is not implemented while actual_seq_lengths is Tensor!")

    return ge.PromptFlashAttention(query, key, value, padding_mask=padding_mask, atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths, num_heads=num_heads, scale_value=scale_value,
        pre_tokens=pre_tokens, next_tokens=next_tokens, input_layout=input_layout,
        num_key_value_heads=num_key_value_heads)


@declare_supported(
    [
        # 支持输入q、k、v，BSH三维格式,q:b=1, s=1, h=40*128;k/v:b=1, s=2048, h=40*128;
        Support(F16(1, 1, 40 * 128), F16(1, 2048, 40 * 128), F16(1, 2048, 40 * 128),
            num_heads=40, input_layout="BSH"),
        # 支持输入q、k、v，BNSD四维格式
        Support(F16(1, 40, 1, 128), F16(1, 40, 2048, 128), F16(1, 40, 2048, 128),
            num_heads=40, input_layout="BNSD"),
        # 支持设置scale_value
        Support(F16(1, 40, 1, 128), F16(1, 40, 2048, 128), F16(1, 40, 2048, 128),
            input_layout="BNSD", num_heads=40, scale_value=0.0884),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_incre_flash_attention.default)
def convert_npu_npu_incre_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    padding_mask: Optional[Tensor] = None,
    atten_mask: Optional[Tensor] = None,
    actual_seq_lengths: Optional[Union[List[int], Tensor]] = None,
    num_heads: int = 1,
    scale_value: float = 1.0,
    input_layout: str = "BSH",
    num_key_value_heads: int = 0,
    meta_outputs: TensorSpec = None,
):

    '''NB: npu::npu_incre_flash_attention(Tensor query, Tensor key, Tensor value, *, Tensor? padding_mask=None, Tensor? atten_mask=None, int[]? actual_seq_lengths=None, int num_heads=1, float scale_value=1.0, str input_layout="BSH", int num_key_value_heads=0) -> Tensor'''
    if actual_seq_lengths is not None and isinstance(actual_seq_lengths, Tensor):
        raise NotImplementedError("IncreFlashAttention is not implemented while actual_seq_lengths is Tensor!")
    key_list = [key]
    value_list = [value]
    
    return ge.IncreFlashAttention(query, key_list, value_list, padding_mask=padding_mask, atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths, num_heads=num_heads, scale_value=scale_value,
        input_layout=input_layout, num_key_value_heads=num_key_value_heads)
