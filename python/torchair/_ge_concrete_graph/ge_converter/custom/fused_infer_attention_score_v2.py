from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        # 支持输入q、k、v，BSH三维格式,q:b=1, s=2048, h=40*128;k/v:b=1, s=2048, h=40*128;
        Support(F16(1, 2048, 40 * 128), F16(1, 2048, 40 * 128), F16(1, 2048, 40 * 128),
            num_query_heads=40, input_layout="BSH"),
        # 支持输入q、k、v，BNSD四维格式
        Support(F16(1, 40, 2048, 128), F16(1, 40, 2048, 128), F16(1, 40, 2048, 128),
            num_query_heads=40, input_layout="BNSD"),
        # 支持设置scale_value
        Support(F16(1, 40, 2048, 128), F16(1, 40, 2048, 128), F16(1, 40, 2048, 128),
            input_layout="BNSD", num_query_heads=40, softmax_scale=0.0884),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_fused_infer_attention_score_v2.default)
def convert_npu_npu_fused_infer_attention_score_v2(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_rope: Optional[Tensor] = None,
    key_rope: Optional[Tensor] = None,
    pse_shift: Optional[Tensor] = None,
    atten_mask: Optional[Tensor] = None,
    actual_seq_qlen: Optional[Union[List[int], Tensor]] = None,
    actual_seq_kvlen: Optional[Union[List[int], Tensor]] = None,
    block_table: Optional[Tensor] = None,
    dequant_scale_query: Optional[Tensor] = None,
    dequant_scale_key: Optional[Tensor] = None,
    dequant_offset_key: Optional[Tensor] = None,
    dequant_scale_value: Optional[Tensor] = None,
    dequant_offset_value: Optional[Tensor] = None,
    dequant_scale_key_rope: Optional[Tensor] = None,
    quant_scale_out: Optional[Tensor] = None,
    quant_offset_out: Optional[Tensor] = None,
    num_query_heads: int = 1,
    num_key_value_heads: int = 0,
    softmax_scale: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    input_layout: str = "BSH",
    sparse_mode: int = 0,
    block_size: int = 0,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    value_quant_mode: int = 0,
    inner_precise: int = 0,
    return_softmax_lse: bool = False,
    query_dtype: Optional[int] = None,
    key_dtype: Optional[int] = None,
    value_dtype: Optional[int] = None,
    query_rope_dtype: Optional[int] = None,
    key_rope_dtype: Optional[int] = None,
    key_shared_prefix_dtype: Optional[int] = None,
    value_shared_prefix_dtype: Optional[int] = None,
    dequant_scale_query_dtype: Optional[int] = None,
    dequant_scale_key_dtype: Optional[int] = None,
    dequant_scale_value_dtype: Optional[int] = None,
    dequant_scale_key_rope_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    # 禁止单独修改此函数，请同步修改传device tensor的actual seq length接口
    if input_layout == 'BSH':
        const = ge.Const([1, 1, 8])
    else:
        const = ge.Const([1, 1, 1, 8])
    if key is not None and key.dtype == DataType.DT_INT32:
        shape = ge.Shape(key)
        key_shape = ge.Mul(shape, const)
        key = ge.Bitcast(key, type=DataType.DT_INT4)
        key = ge.Reshape(key, key_shape)

    if value is not None and value.dtype == DataType.DT_INT32:
        shape = ge.Shape(value)
        value_shape = ge.Mul(shape, const)
        value = ge.Bitcast(value, type=DataType.DT_INT4)
        value = ge.Reshape(value, value_shape)

    key_list = [key]
    value_list = [value]
    if actual_seq_qlen is not None:
        actual_seq_qlen = dtype_promote(actual_seq_qlen, target_dtype=DataType.DT_INT64)
    if actual_seq_kvlen is not None:
        actual_seq_kvlen = dtype_promote(actual_seq_kvlen, target_dtype=DataType.DT_INT64)
    # dropped params
    quant_scale1 = None
    dequant_scale2 = None
    dequant_scale1 = None
    antiquant_scale = None
    antiquant_offset = None
    query_padding_size = None
    kv_padding_size = None
    key_shared_prefix = None
    value_shared_prefix = None
    actual_shared_prefix_len = None
    antiquant_mode = 0

    return ge.FusedInferAttentionScore(query, key_list, value_list, pse_shift=pse_shift, atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_qlen, actual_seq_lengths_kv=actual_seq_kvlen,
        dequant_scale1=dequant_scale1, quant_scale1=quant_scale1, dequant_scale2=dequant_scale2,
        quant_scale2=quant_scale_out, quant_offset2=quant_offset_out, antiquant_scale=antiquant_scale,
        antiquant_offset=antiquant_offset, block_table=block_table, query_padding_size=query_padding_size,
        kv_padding_size=kv_padding_size, key_antiquant_scale=dequant_scale_key,
        key_antiquant_offset=dequant_offset_key, value_antiquant_scale=dequant_scale_value,
        value_antiquant_offset=dequant_offset_value, key_shared_prefix=key_shared_prefix,
        value_shared_prefix=value_shared_prefix, actual_shared_prefix_len=actual_shared_prefix_len,
        query_rope=query_rope, key_rope=key_rope, key_rope_antiquant_scale=dequant_scale_key_rope,
        dequant_scale_query=dequant_scale_query, num_heads=num_query_heads, scale=softmax_scale,
        pre_tokens=pre_tokens, next_tokens=next_tokens, input_layout=input_layout,
        num_key_value_heads=num_key_value_heads, sparse_mode=sparse_mode, inner_precise=inner_precise,
        block_size=block_size, antiquant_mode=antiquant_mode, softmax_lse_flag=return_softmax_lse,
        key_antiquant_mode=key_quant_mode, value_antiquant_mode=value_quant_mode, query_quant_mode=query_quant_mode)