from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload, Dict,
)

import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._utils.error_code import pretty_error_msg

lib = torch.library.Library("air", "FRAGMENT")
lib.define(
    """
    npu_fused_infer_attention_score(Tensor query, Tensor key, Tensor value, *, Tensor? pse_shift=None, \
    Tensor? atten_mask=None, Tensor? actual_seq_lengths=None, Tensor? actual_seq_lengths_kv=None, \
    Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, \
    Tensor? quant_offset2=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, \
    Tensor? key_antiquant_scale=None, Tensor? key_antiquant_offset=None, Tensor? value_antiquant_scale=None, \
    Tensor? value_antiquant_offset=None, Tensor? block_table=None, Tensor? query_padding_size=None, \
    Tensor? kv_padding_size=None, Tensor? key_shared_prefix=None, Tensor? value_shared_prefix=None, \
    Tensor? actual_shared_prefix_len=None, Tensor? query_rope=None, Tensor? key_rope=None, \
    Tensor? key_rope_antiquant_scale=None, int num_heads=1, float scale=1.0, int pre_tokens=2147483647, \
    int next_tokens=2147483647, str input_layout="BSH", int num_key_value_heads=0, int sparse_mode=0, \
    int inner_precise=0, int block_size=0, int antiquant_mode=0, int key_antiquant_mode=0, int value_antiquant_mode=0, \
    bool softmax_lse_flag=False) -> (Tensor, Tensor)
    """
)


def _npu_fused_infer_attention_score(*args, **kwargs):
    return torch.ops.air.npu_fused_infer_attention_score(*args, **kwargs)


def npu_fused_infer_attention_score_impl(*args, **kwargs):
    raise NotImplementedError("eager mode of torchair.ops.npu_fused_infer_attention_score is not supported, " +
                              "use graph mode or torch_npu.npu_fused_infer_attention_score!")


torch.library.impl(lib, "npu_fused_infer_attention_score", "CPU")(npu_fused_infer_attention_score_impl)
torch.library.impl(lib, "npu_fused_infer_attention_score", "PrivateUse1")(npu_fused_infer_attention_score_impl)


@declare_supported(
    [
        # 支持输入q、k、v，BSH三维格式, q:b=1, s=2048, h=40*128;k/v:b=1, s=2048, h=40*128;
        Support(F16(1, 2048, 40 * 128), F16(1, 2048, 40 * 128), F16(1, 2048, 40 * 128),
            num_heads=40, input_layout="BSH"),
        # 支持输入q、k、v，BNSD四维格式
        Support(F16(1, 40, 2048, 128), F16(1, 40, 2048, 128), F16(1, 40, 2048, 128),
            num_heads=40, input_layout="BNSD"),
        # 支持设置scale_value
        Support(F16(1, 40, 2048, 128), F16(1, 40, 2048, 128), F16(1, 40, 2048, 128),
            input_layout="BNSD", num_heads=40, scale=0.0884),
    ]
)
@register_fx_node_ge_converter(torch.ops.air.npu_fused_infer_attention_score.default)
def convert_npu_npu_fused_infer_attention_score_tensor(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    pse_shift: Optional[Tensor] = None,
    atten_mask: Optional[Tensor] = None,
    actual_seq_lengths: Optional[Tensor] = None,
    actual_seq_lengths_kv: Optional[Tensor] = None,
    dequant_scale1: Optional[Tensor] = None,
    quant_scale1: Optional[Tensor] = None,
    dequant_scale2: Optional[Tensor] = None,
    quant_scale2: Optional[Tensor] = None,
    quant_offset2: Optional[Tensor] = None,
    antiquant_scale: Optional[Tensor] = None,
    antiquant_offset: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    query_padding_size: Optional[Tensor] = None,
    kv_padding_size: Optional[Tensor] = None,
    key_antiquant_scale: Optional[Tensor] = None,
    key_antiquant_offset: Optional[Tensor] = None,
    value_antiquant_scale: Optional[Tensor] = None,
    value_antiquant_offset: Optional[Tensor] = None,
    key_shared_prefix: Optional[Tensor] = None,
    value_shared_prefix: Optional[Tensor] = None,
    actual_shared_prefix_len: Optional[Tensor] = None,
    query_rope: Optional[Tensor] = None,
    key_rope: Optional[Tensor] = None,
    key_rope_antiquant_scale: Optional[Tensor] = None,
    num_heads: int = 1,
    scale: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    input_layout: str = "BSH",
    num_key_value_heads: int = 0,
    sparse_mode: int = 0,
    inner_precise: int = 0,
    block_size: int = 0,
    antiquant_mode: int = 0,
    softmax_lse_flag: bool = False,
    key_antiquant_mode: int = 0,
    value_antiquant_mode: int = 0,
    meta_outputs: TensorSpec = None,
):
    # 禁止单独修改此函数，请同步修改actual seq length为symint list的接口
    if input_layout == 'BSH':
        const = ge.Const([1, 1, 8])
    else:
        const = ge.Const([1, 1, 1, 8])
    if key is not None and key.dtype == DataType.DT_INT32:
        shape = ge.Shape(key)
        key_shape = ge.Mul(shape, const)
        key = ge.Bitcast(key, type=DataType.DT_INT4)
        key = ge.Reshape(key, key_shape)

    if key_shared_prefix is not None and key_shared_prefix.dtype == DataType.DT_INT32:
        shape = ge.Shape(key_shared_prefix)
        key_shared_prefix_shape = ge.Mul(shape, const)
        key_shared_prefix = ge.Bitcast(key_shared_prefix, type=DataType.DT_INT4)
        key_shared_prefix = ge.Reshape(key_shared_prefix, key_shared_prefix_shape)

    if value is not None and value.dtype == DataType.DT_INT32:
        shape = ge.Shape(value)
        value_shape = ge.Mul(shape, const)
        value = ge.Bitcast(value, type=DataType.DT_INT4)
        value = ge.Reshape(value, value_shape)

    if value_shared_prefix is not None and value_shared_prefix.dtype == DataType.DT_INT32:
        shape = ge.Shape(value_shared_prefix)
        value_shared_prefix_shape = ge.Mul(shape, const)
        value_shared_prefix = ge.Bitcast(value_shared_prefix, type=DataType.DT_INT4)
        value_shared_prefix = ge.Reshape(value_shared_prefix, value_shared_prefix_shape)

    key_list = [key]
    value_list = [value]
    dequant_scale_query = None
    learnable_sink = None
    query_quant_mode = 0
    return ge.FusedInferAttentionScore(query, key_list, value_list, pse_shift=pse_shift, atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths, actual_seq_lengths_kv=actual_seq_lengths_kv,
        dequant_scale1=dequant_scale1, quant_scale1=quant_scale1, dequant_scale2=dequant_scale2,
        quant_scale2=quant_scale2, quant_offset2=quant_offset2, antiquant_scale=antiquant_scale,
        antiquant_offset=antiquant_offset, block_table=block_table, query_padding_size=query_padding_size,
        kv_padding_size=kv_padding_size, key_antiquant_scale=key_antiquant_scale,
        key_antiquant_offset=key_antiquant_offset, value_antiquant_scale=value_antiquant_scale,
        value_antiquant_offset=value_antiquant_offset, key_shared_prefix=key_shared_prefix,
        value_shared_prefix=value_shared_prefix, actual_shared_prefix_len=actual_shared_prefix_len,
        query_rope=query_rope, key_rope=key_rope, key_rope_antiquant_scale=key_rope_antiquant_scale,
        dequant_scale_query=dequant_scale_query, learnable_sink=learnable_sink, q_start_idx=None, kv_start_idx=None,
        num_heads=num_heads, scale=scale, pre_tokens=pre_tokens, next_tokens=next_tokens, input_layout=input_layout, 
        num_key_value_heads=num_key_value_heads, sparse_mode=sparse_mode, inner_precise=inner_precise, 
        block_size=block_size, antiquant_mode=antiquant_mode, softmax_lse_flag=softmax_lse_flag, 
        key_antiquant_mode=key_antiquant_mode, value_antiquant_mode=value_antiquant_mode, 
        query_quant_mode=query_quant_mode, pse_type=0, out_dtype=0)


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


@torch.library.impl(lib, "npu_fused_infer_attention_score", "Meta")
def npu_fused_infer_attention_score_meta_impl(query, key, value, *, pse_shift=None, atten_mask=None,
    actual_seq_lengths=None, actual_seq_lengths_kv=None, dequant_scale1=None, quant_scale1=None, dequant_scale2=None,
    quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None, block_table=None,
    query_padding_size=None, kv_padding_size=None, key_antiquant_scale=None, key_antiquant_offset=None,
    value_antiquant_scale=None, value_antiquant_offset=None, key_shared_prefix=None, value_shared_prefix=None,
    actual_shared_prefix_len=None, query_rope=None, key_rope=None, key_rope_antiquant_scale=None, num_heads=1,
    scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", num_key_value_heads=0, sparse_mode=0,
    inner_precise=0, block_size=0, antiquant_mode=0, softmax_lse_flag=False, key_antiquant_mode=0,
    value_antiquant_mode=0):
    # 禁止单独修改此函数，请同步修改actual seq length为symint list的接口
    if num_heads <= 0:
        raise ValueError(
            f'numHeads should be greater than 0, but got {num_heads}!')
    if num_key_value_heads == 0:
        num_key_value_heads = num_heads

    query_layout, attention_out_layout = get_query_and_attention_out_layout(query, input_layout)

    value_d = get_value_d(block_table, value, query, query_layout, num_key_value_heads)

    tmp_out = infer_attention_out_shape(attention_out_layout, query, query_layout, num_heads, value_d)

    # special:IFA legacy feature
    change_d_scale = 1
    if value is not None and value.dtype == DataType.DT_INT32: # treat int4 as int32
        change_d_scale = 8
    if input_layout == "BNSD" and block_table is None:
        tmp_out = torch.empty([query.size(0), query.size(1), query.size(2), value.size(3) * change_d_scale],
            dtype=query.dtype, device='meta')

    if quant_scale2 is not None:
        attention_out = torch.empty_like(tmp_out, dtype=torch.int8)
    elif query.dtype == torch.int8:
        if query_rope is not None:
            attention_out = torch.empty_like(tmp_out, dtype=query_rope.dtype)
        else:
            attention_out = torch.empty_like(tmp_out, dtype=torch.half)
    else:
        attention_out = torch.empty_like(tmp_out, dtype=query.dtype)

    tmp_lse_out = infer_lse_out_shape(query, input_layout, query_layout, num_heads)

    if softmax_lse_flag:
        lse_out = torch.empty_like(tmp_lse_out, dtype=torch.float32)
    else:
        lse_out = torch.empty([0], dtype=torch.float32, device='meta')

    return attention_out, lse_out