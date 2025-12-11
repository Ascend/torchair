from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload, Dict,
)

import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, ge_type_to_torch_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._utils.error_code import pretty_error_msg

lib = torch.library.Library("air", "FRAGMENT")
lib.define(
    """
    npu_fused_infer_attention_score_v2(Tensor query, Tensor key, Tensor value, *, Tensor? query_rope=None, \
    Tensor? key_rope=None, Tensor? pse_shift=None, Tensor? atten_mask=None, Tensor? actual_seq_qlen=None,  \
    Tensor? actual_seq_kvlen=None, Tensor? block_table=None, Tensor? dequant_scale_query=None, \
    Tensor? dequant_scale_key=None, Tensor? dequant_offset_key=None, Tensor? dequant_scale_value=None,  \
    Tensor? dequant_offset_value=None, Tensor? dequant_scale_key_rope=None,  \
    Tensor? quant_scale_out=None, Tensor? quant_offset_out=None, Tensor? learnable_sink=None, int num_query_heads=1, \
    int num_key_value_heads=0,  float softmax_scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, \
    str input_layout="BSH", int sparse_mode=0, int block_size=0, int query_quant_mode=0, int key_quant_mode=0, \
    int value_quant_mode=0, int inner_precise=0, bool return_softmax_lse=False, int? query_dtype=None, \
    int? key_dtype=None, int? value_dtype=None, int? query_rope_dtype=None, int? key_rope_dtype=None, \
    int? key_shared_prefix_dtype=None, int? value_shared_prefix_dtype=None, int? dequant_scale_query_dtype=None, \
    int? dequant_scale_key_dtype=None, int? dequant_scale_value_dtype=None, int? dequant_scale_key_rope_dtype=None, \
    int? out_dtype=None) \
    -> (Tensor, Tensor)
    """
)


def _npu_fused_infer_attention_score_v2(*args, **kwargs):
    return torch.ops.air.npu_fused_infer_attention_score_v2(*args, **kwargs)


def npu_fused_infer_attention_score_v2_impl(*args, **kwargs):
    raise NotImplementedError("torchair.ops.npu_fused_infer_attention_score_v2 does not support eager mode or reduce-overhead mode, " +
                              "support using max-autotune mode or torch_npu.npu_fused_infer_attention_score_v2!")


torch.library.impl(lib, "npu_fused_infer_attention_score_v2", "CPU")(npu_fused_infer_attention_score_v2_impl)
torch.library.impl(lib, "npu_fused_infer_attention_score_v2", "PrivateUse1")(npu_fused_infer_attention_score_v2_impl)


@declare_supported(
    [
        # 支持输入q、k、v，BSH三维格式, q:b=1, s=2048, h=40*128;k/v:b=1, s=2048, h=40*128;
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
@register_fx_node_ge_converter(torch.ops.air.npu_fused_infer_attention_score_v2.default)
def convert_npu_npu_fused_infer_attention_score_v2_tensor(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_rope: Optional[Tensor] = None,
    key_rope: Optional[Tensor] = None,
    pse_shift: Optional[Tensor] = None,
    atten_mask: Optional[Tensor] = None,
    actual_seq_qlen: Optional[Tensor] = None,
    actual_seq_kvlen: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    dequant_scale_query: Optional[Tensor] = None,
    dequant_scale_key: Optional[Tensor] = None,
    dequant_offset_key: Optional[Tensor] = None,
    dequant_scale_value: Optional[Tensor] = None,
    dequant_offset_value: Optional[Tensor] = None,
    dequant_scale_key_rope: Optional[Tensor] = None,
    quant_scale_out: Optional[Tensor] = None,
    quant_offset_out: Optional[Tensor] = None,
    learnable_sink: Optional[Tensor] = None,
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
    out_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    # 禁止单独修改此函数，请同步修改actual seq length为symint list的接口
    try:
        import torch_npu
    except Exception as e:
        raise RuntimeError(f"{e} while checking value data type") from e
    is_int4 = (key is not None and key.dtype == DataType.DT_INT32) or (value is not None and value.dtype == DataType.DT_INT32)
    is_fp4 = key_dtype == torch_npu.float4_e2m1fn_x2 or value_dtype == torch_npu.float4_e2m1fn_x2 or key_dtype == torch_npu.float4_e1m2fn_x2 or value_dtype == torch_npu.float4_e1m2fn_x2
    if is_int4 or is_fp4:
        shape_multiples = 1
        key_ge_dtype = 0
        value_ge_dtype = 0
        if is_int4:
            shape_multiples = 8
            key_ge_dtype = DataType.DT_INT4
            value_ge_dtype = DataType.DT_INT4
        elif is_fp4:
            shape_multiples = 2
            key_ge_dtype = torch_dtype_value_to_ge_type(key_dtype)
            value_ge_dtype = torch_dtype_value_to_ge_type(value_dtype)
        if input_layout == 'BSH' or key.rank == 3 or value.rank == 3:
            const = ge.Const([1, 1, shape_multiples])
            if key.rank == 5 or value.rank == 5:
                const = ge.Const([1, 1, 1, 1, shape_multiples])
        else:
            const = ge.Const([1, 1, 1, shape_multiples])
            if key.rank == 5 or value.rank == 5:
                const = ge.Const([1, 1, 1, 1, shape_multiples])
        if key is not None:
            shape = ge.Shape(key)
            key_shape = ge.Mul(shape, const)
            key = ge.Bitcast(key, type=key_ge_dtype)
            key = ge.Reshape(key, key_shape)
        if value is not None:
            shape = ge.Shape(value)
            value_shape = ge.Mul(shape, const)
            value = ge.Bitcast(value, type=value_ge_dtype)
            value = ge.Reshape(value, value_shape)

        if dequant_scale_key is not None and dequant_scale_key_dtype == torch_npu.float8_e8m0fnu:
            dequant_scale_key_ge_dtype = torch_dtype_value_to_ge_type(dequant_scale_key_dtype)
            dequant_scale_key = ge.Bitcast(dequant_scale_key, type=dequant_scale_key_ge_dtype)
        if dequant_scale_value is not None and dequant_scale_value_dtype == torch_npu.float8_e8m0fnu:
            dequant_scale_value_ge_dtype = torch_dtype_value_to_ge_type(dequant_scale_value_dtype)
            dequant_scale_value = ge.Bitcast(dequant_scale_value, type=dequant_scale_value_ge_dtype)

    if key is not None and key_dtype == torch_npu.hifloat8:
        key = ge.Bitcast(key, type=DataType.DT_HIFLOAT8)
    if value is not None and value_dtype == torch_npu.hifloat8:
        value = ge.Bitcast(value, type=DataType.DT_HIFLOAT8)

    key_list = [key]
    value_list = [value]
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
        dequant_scale_query=dequant_scale_query, learnable_sink=learnable_sink,
        q_start_idx=None, kv_start_idx=None, num_heads=num_query_heads, scale=softmax_scale,
        pre_tokens=pre_tokens, next_tokens=next_tokens, input_layout=input_layout,
        num_key_value_heads=num_key_value_heads, sparse_mode=sparse_mode, inner_precise=inner_precise,
        block_size=block_size, antiquant_mode=antiquant_mode, softmax_lse_flag=return_softmax_lse,
        key_antiquant_mode=key_quant_mode, value_antiquant_mode=value_quant_mode, query_quant_mode=query_quant_mode,
        pse_type=0, out_dtype=out_dtype)


def get_query_and_attention_out_layout(query, input_layout):
    class ParserLayout:
        def __init__(self, qLayout: str, outLayout: str, qDim: int):
            self.qLayout = qLayout
            self.outLayout = outLayout
            self.qDim = qDim

    LAYOUT_MAP: Dict[str, ParserLayout] = {
        "BSH": ParserLayout("BSH", "BSH", 3),
        "BSND": ParserLayout("BSND", "BSND", 4),
        "BNSD": ParserLayout("BNSD", "BNSD", 4),
        "TND": ParserLayout("TND", "TND", 3),
        "NTD": ParserLayout("NTD", "NTD", 3),
        "BNSD_BSND": ParserLayout("BNSD", "BSND", 4),
        "BSH_BNSD": ParserLayout("BSH", "BNSD", 3),
        "BSND_BNSD": ParserLayout("BSND", "BNSD", 4),
        "NTD_TND": ParserLayout("NTD", "TND", 3),
        "BSH_NBSD": ParserLayout("BSH", "NBSD", 3),
        "BSND_NBSD": ParserLayout("BSND", "NBSD", 4),
        "BNSD_NBSD": ParserLayout("BNSD", "NBSD", 4),
        "TND_NTD": ParserLayout("TND", "NTD", 3),
        "NSD": ParserLayout("NSD", "NSD", 3)
    }

    if input_layout in LAYOUT_MAP:
        layout_entry = LAYOUT_MAP[input_layout]

        query_layout = layout_entry.qLayout
        attention_out_layout = layout_entry.outLayout
        query_dim = layout_entry.qDim

        if query.dim() != query_dim:
            raise ValueError(
                f'Layout {query_layout}, queryDims({query.dim()}) must be {query_dim}!')
    else:
        raise ValueError(
            f'Layout {input_layout} is not supported!')
    return query_layout, attention_out_layout


def get_query_b_n_s(query, query_layout, num_heads):
    if query_layout == "BSH":
        b = query.size(0)
        s1 = query.size(1)
        n1 = num_heads
    elif query_layout == "BSND":
        b = query.size(0)
        s1 = query.size(1)
        n1 = query.size(2)
    elif query_layout == "BNSD":
        b = query.size(0)
        s1 = query.size(2)
        n1 = query.size(1)
    elif query_layout == "NSD":
        b = 1
        s1 = query.size(1)
        n1 = query.size(0)
    else:
        raise ValueError(
            f'Layout {query_layout} is not supported in get_query_b_n_s function!')
    return b, s1, n1


def get_query_t_n(query, query_layout):
    if query_layout == "TND":
        t = query.size(0)
        n1 = query.size(1)
    elif query_layout == "NTD":
        t = query.size(1)
        n1 = query.size(0)
    else:
        raise ValueError(
            f'Layout {query_layout} is not supported in get_query_t_n function!')
    return t, n1


def get_value_d(block_table, value, query, query_layout, num_kv_heads):
    if block_table is not None:
        if value.dim() == 3:
            value_d = value.size(2) // num_kv_heads
        elif value.dim() == 4:
            value_d = value.size(3)
        elif value.dim() == 5:
            value_d = value.size(2) * value.size(4)
        else:
            raise ValueError(
                f'when Page Attention enabled, value dim should be 3/4/5, but got {value.dim()}!')
    else:
        if value.dim() != query.dim():
            raise ValueError(
                f'when Page Attention not enabled, value dim{value.dim()} should equal to query dim{query.dim()}!')
        if query_layout == "BSH":
            value_d = value.size(2) // num_kv_heads
        if query_layout == "BNSD" or query_layout == "BSND":
            value_d = value.size(3)
        if query_layout == "TND" or query_layout == "NTD" or query_layout == "NSD":
            value_d = value.size(2)
    return value_d


def get_change_d_scale_v2(value, value_dtype):
    try:
        import torch_npu
    except Exception as e:
        raise RuntimeError(f"{e} while checking value data type") from e
    change_d_scale = 1

    if value is None:
        return change_d_scale
    #int4伪装int32
    if value.dtype == torch.int32:
        change_d_scale = 8
    # value_dtype float4_e2m1fn_x2 伪装 uint8
    if (hasattr(torch, 'float4_e2m1fn_x2') and value.dtype == torch.float4_e2m1fn_x2) or value_dtype == torch_npu.float4_e2m1fn_x2:
        change_d_scale = 2
    # value_dtype float4_e1m2fn_x2 伪装 uint8
    if (hasattr(torch, 'float4_e1m2fn_x2') and value.dtype == torch.float4_e1m2fn_x2) or value_dtype == torch_npu.float4_e1m2fn_x2:
        change_d_scale = 2
    
    return change_d_scale


def infer_attention_out_shape(attention_out_layout, query, query_layout, num_heads, value_d):
    attention_out = torch.empty_like(query, dtype=query.dtype, device='meta')
    if attention_out_layout == "BSH":
        b, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        attention_out = torch.empty([b, s1, n1 * value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "BSND":
        b, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        attention_out = torch.empty([b, s1, n1, value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "BNSD":
        b, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        attention_out = torch.empty([b, n1, s1, value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "NBSD":
        b, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        attention_out = torch.empty([n1, b, s1, value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "TND":
        t, n1 = get_query_t_n(query, query_layout)
        attention_out = torch.empty([t, n1, value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "NTD":
        t, n1 = get_query_t_n(query, query_layout)
        attention_out = torch.empty([n1, t, value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "NSD":
        b, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        attention_out = torch.empty([n1, s1, value_d], dtype=query.dtype, device='meta')
    return attention_out


def infer_lse_out_shape(query, input_layout, query_layout, num_heads):
    lse_out = torch.empty([0], dtype=torch.float32, device='meta')

    tnd_like_layouts = {"TND", "NTD", "TND_NTD", "NTD_TND"}
    if input_layout in tnd_like_layouts:
        t, n1 = get_query_t_n(query, query_layout)
        lse_out = torch.empty([t, n1, 1], dtype=torch.float32, device='meta')
    else:
        b, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        lse_out = torch.empty([b, n1, s1, 1], dtype=torch.float32, device='meta')
    return lse_out


@torch.library.impl(lib, "npu_fused_infer_attention_score_v2", "Meta")
def npu_fused_infer_attention_score_v2_meta_impl(query, key, value, *, query_rope=None, key_rope=None, pse_shift=None,
    atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None, block_table=None, dequant_scale_query=None,
    dequant_scale_key=None, dequant_offset_key=None, dequant_scale_value=None, dequant_offset_value=None,
    dequant_scale_key_rope=None, quant_scale_out=None, quant_offset_out=None, learnable_sink=None, num_query_heads=1, 
    num_key_value_heads=0, softmax_scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", 
    sparse_mode=0, block_size=0, query_quant_mode=0, key_quant_mode=0, value_quant_mode=0, inner_precise=0, 
    return_softmax_lse=False, query_dtype=None, key_dtype=None, value_dtype=None, query_rope_dtype=None, 
    key_rope_dtype=None, key_shared_prefix_dtype=None, value_shared_prefix_dtype=None, dequant_scale_query_dtype=None, 
    dequant_scale_key_dtype=None, dequant_scale_value_dtype=None, dequant_scale_key_rope_dtype=None, out_dtype=None):
    # 禁止单独修改此函数，请同步修改actual seq length为symint list的接口
    if num_query_heads <= 0:
        raise ValueError(
            f'numHeads should be greater than 0, but got {num_query_heads}!')
    if num_key_value_heads == 0:
        num_key_value_heads = num_query_heads

    query_layout, attention_out_layout = get_query_and_attention_out_layout(query, input_layout)

    value_d = get_value_d(block_table, value, query, query_layout, num_key_value_heads)

    change_d_scale = get_change_d_scale_v2(value, value_dtype)
    value_d = value_d * change_d_scale

    tmp_out = infer_attention_out_shape(attention_out_layout, query, query_layout, num_query_heads, value_d)

    if quant_scale_out is not None:
        output_type = torch.int8
        if out_dtype is not None:
            output_type = ge_type_to_torch_type(torch_dtype_value_to_ge_type(out_dtype))
        attention_out = torch.empty_like(tmp_out, dtype=output_type)
    elif query.dtype == torch.int8 or query.dtype == torch.float8_e4m3fn:
        if query_rope is not None:
            attention_out = torch.empty_like(tmp_out, dtype=query_rope.dtype)
        else:
            attention_out = torch.empty_like(tmp_out, dtype=torch.half)
    else:
        attention_out = torch.empty_like(tmp_out, dtype=query.dtype)

    tmp_lse_out = infer_lse_out_shape(query, input_layout, query_layout, num_query_heads)

    if return_softmax_lse:
        lse_out = torch.empty_like(tmp_lse_out, dtype=torch.float32)
    else:
        lse_out = torch.empty([0], dtype=torch.float32, device='meta')

    return attention_out, lse_out