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
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, BF16, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


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
    pse_shift: Optional[Tensor] = None,
    actual_seq_lengths: Optional[Union[List[int], Tensor]] = None,
    deq_scale1: Optional[Tensor] = None,
    quant_scale1: Optional[Tensor] = None,
    deq_scale2: Optional[Tensor] = None,
    quant_scale2: Optional[Tensor] = None,
    quant_offset2: Optional[Tensor] = None,
    num_heads: int = 1,
    scale_value: float = 1.0,
    pre_tokens: int = 2147473647,
    next_tokens: int = 0,
    input_layout: str = "BSH",
    num_key_value_heads: int = 0,
    actual_seq_lengths_kv: Optional[Union[List[int], Tensor]] = None,
    sparse_mode: int = 0,
    inner_precise: int = 1,
    meta_outputs: TensorSpec = None,
):

    '''NB: npu::npu_prompt_flash_attention(Tensor query, Tensor key, Tensor value, *, Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? pse_shift=None, int[]? actual_seq_lengths=None, int num_heads=1, float scale_value=1.0, int pre_tokens=2147473647, int next_tokens=0, str input_layout="BSH", int num_key_value_heads=0, int[]? actual_seq_lengths_kv=None, int sparse_mode=0, int inner_precise=1) -> Tensor'''
    if actual_seq_lengths is not None and isinstance(actual_seq_lengths, Tensor):
        raise NotImplementedError("PromptFlashAttention is not implemented while actual_seq_lengths is Tensor!")
    if actual_seq_lengths_kv is not None and isinstance(actual_seq_lengths_kv, Tensor):
        raise NotImplementedError("PromptFlashAttention is not implemented while actual_seq_lengths_kv is Tensor!")
    if actual_seq_lengths is not None:
        actual_seq_lengths = dtype_promote(actual_seq_lengths, target_dtype=DataType.DT_INT64)
    if actual_seq_lengths_kv is not None:
        actual_seq_lengths_kv = dtype_promote(actual_seq_lengths_kv, target_dtype=DataType.DT_INT64)

    if sparse_mode >= 10 and sparse_mode <= 14:
        inner_precise = 0
        sparse_mode -= 10

    return ge.PromptFlashAttention(query, key, value, pse_shift=pse_shift, atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths, actual_seq_lengths_kv=actual_seq_lengths_kv,
        deq_scale1=deq_scale1, quant_scale1=quant_scale1, deq_scale2=deq_scale2,
        quant_scale2=quant_scale2, quant_offset2=quant_offset2,
        num_heads=num_heads, scale_value=scale_value,
        pre_tokens=pre_tokens, next_tokens=next_tokens, input_layout=input_layout,
        num_key_value_heads=num_key_value_heads, sparse_mode=sparse_mode, inner_precise=inner_precise)


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
    pse_shift: Optional[Tensor] = None,
    actual_seq_lengths: Optional[Union[List[int], Tensor]] = None,
    antiquant_scale: Optional[Tensor] = None,
    antiquant_offset: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    dequant_scale1: Optional[Tensor] = None,
    quant_scale1: Optional[Tensor] = None,
    dequant_scale2: Optional[Tensor] = None,
    quant_scale2: Optional[Tensor] = None,
    quant_offset2: Optional[Tensor] = None,
    kv_padding_size: Optional[Tensor] = None,
    num_heads: int = 1,
    scale_value: float = 1.0,
    input_layout: str = "BSH",
    num_key_value_heads: int = 0,
    block_size: int = 0,
    inner_precise: int = 1,
    meta_outputs: TensorSpec = None,
):

    '''NB: npu_incre_flash_attention(Tensor query, Tensor key, Tensor value, *, Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? pse_shift=None, SymInt[]? actual_seq_lengths=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? block_table=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? kv_padding_size=None, int num_heads=1, float scale_value=1.0, str input_layout="BSH", int num_key_value_heads=0, int block_size=0, int inner_precise=1) -> Tensor'''
    key_list = [key]
    value_list = [value]
    if actual_seq_lengths is not None:
        actual_seq_lengths = dtype_promote(actual_seq_lengths, target_dtype=DataType.DT_INT64)

    return ge.IncreFlashAttention(query, key_list, value_list, pse_shift=pse_shift, atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths, dequant_scale1=dequant_scale1, quant_scale1=quant_scale1,
        dequant_scale2=dequant_scale2, quant_scale2=quant_scale2, quant_offset2=quant_offset2,
        antiquant_scale=antiquant_scale, antiquant_offset=antiquant_offset, block_table=block_table,
        kv_padding_size=kv_padding_size, num_heads=num_heads, scale_value=scale_value, input_layout=input_layout,
        num_key_value_heads=num_key_value_heads, block_size=block_size, inner_precise=inner_precise)


@declare_supported(
    [
        # 支持mla MMQcQr kvcache量化,token_x:B=1, s=1, He=7168; weight_dq:He=7168, Hcq=1536;
        # weight_uq_qr:Hcq=1536, N*(D+Dr)=(128+64)*128; weight_uk:N=128, D=128, Hckv=512;
        # weight_dkv_kr:He=7168, Hckv+Dr=512+64; rms_norm_weight:Hcq=1536;
        # rmsnorm_gamma_ckv:Hckv=512; rope_sin:B=1, S=1, Dr=64; rope_cos:B=1, S=1, Dr=64;
        # cache_index:B=1, S=1; kv_cache:B=1, Nkv=1, Skv=288, Hckv=512;
        # kr_cache:B=1, Nkv=1, Skv=288, Dr=64; dequant_scale_w_uq_qr:1, N*(D+Dr)=(128+64)*128;
        # quant_scale_ckv:1, Hckv=512;
        # quant_scale_ckr:1, Dr=64; smooth_scales_cq:1, Hcq=1536;
        Support(BF16(1, 1, 7168), BF16(7168, 1536),
                I8(1536, 6144), BF16(128, 128, 512),
                BF16(7168, 576), BF16(1536),
                BF16(512), BF16(1, 1, 64), BF16(1, 1, 64),
                I64(1, 1), I8(1, 1, 288, 512),
                I8(1, 1, 288, 64), F32(1, 6144),
                F32(1, 512),
                F32(1, 64), F32(1, 1536),
                rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5),
        # 支持mla MMQcQr kvcache量化,BS合轴,token_x:T=1, He=7168; weight_dq:He=7168, Hcq=1536;
        # weight_uq_qr:Hcq=1536, N*(D+Dr)=(128+64)*128; weight_uk:N=128, D=128, Hckv=512;
        # weight_dkv_kr:He=7168, Hckv+Dr=512+64; rms_norm_weight:Hcq=1536;
        # rmsnorm_gamma_ckv:Hckv=512; rope_sin:T=1, Dr=64; rope_cos:T=1, Dr=64;
        # cache_index:T=1; kv_cache:B=1, Nkv=1, Skv=288, Hckv=512;
        # kr_cache:B=1, Nkv=1, Skv=288, Dr=64; dequant_scale_w_uq_qr:1, N*(D+Dr)=(128+64)*128;
        # quant_scale_ckv:1, Hckv=512;
        # quant_scale_ckr:1, Dr=64; smooth_scales_cq:1, Hcq=1536;
        Support(BF16(1, 7168), BF16(7168, 1536),
                I8(1536, 6144), BF16(128, 128, 512),
                BF16(7168, 576), BF16(1536),
                BF16(512), BF16(1, 64), BF16(1, 64),
                I64(1), I8(1, 1, 288, 512),
                I8(1, 1, 288, 64), F32(1, 6144),
                F32(1, 512),
                F32(1, 64), F32(1, 1536),
                rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5),
        # 支持mla MMQcQr kvcache非量化,token_x:B=1, s=1, He=7168; weight_dq:He=7168, Hcq=1536;
        # weight_uq_qr:Hcq=1536, N*(D+Dr)=(128+64)*128; weight_uk:N=128, D=128, Hckv=512;
        # weight_dkv_kr:He=7168, Hckv+Dr=512+64; rms_norm_weight:Hcq=1536;
        # rmsnorm_gamma_ckv:Hckv=512; rope_sin:B=1, S=1, Dr=64; rope_cos:B=1, S=1, Dr=64;
        # cache_index:B=1, S=1; kv_cache:B=1, Nkv=1, Skv=288, Hckv=512;
        # kr_cache:B=1, Nkv=1, Skv=288, Dr=64; dequant_scale_w_dkv_kr:1, Hckv+Dr=512+64;
        # smooth_scales_cq:1, Hcq=1536;
        Support(BF16(1, 1, 7168), BF16(7168, 1536),
                I8(1536, 6144), BF16(128, 128, 512),
                BF16(7168, 576), BF16(1536),
                BF16(512), BF16(1, 1, 64), BF16(1, 1, 64),
                I64(1, 1), BF16(1, 1, 288, 512),
                BF16(1, 1, 288, 64), F32(1, 6144),
                F32(1, 1536),
                rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5),
        # 支持mla MMQcQr kvcache非量化,BS合轴,token_x:T=1, He=7168; weight_dq:He=7168, Hcq=1536;
        # weight_uq_qr:Hcq=1536, N*(D+Dr)=(128+64)*128; weight_uk:N=128, D=128, Hckv=512;
        # weight_dkv_kr:He=7168, Hckv+Dr=512+64; rms_norm_weight:Hcq=1536;
        # rmsnorm_gamma_ckv:Hckv=512; rope_sin:T=1, Dr=64; rope_cos:T=1, Dr=64;
        # cache_index:T=1;kv_cache:B=1, Nkv=1, Skv=288, Hckv=512;
        # kr_cache:B=1, Nkv=1, Skv=288, Dr=64; dequant_scale_w_dkv_kr:1, Hckv+Dr=512+64;
        # smooth_scales_cq:1, Hcq=1536;
        Support(BF16(1, 7168), BF16(7168, 1536),
                I8(1536, 6144), BF16(128, 128, 512),
                BF16(7168, 576), BF16(1536),
                BF16(512), BF16(1, 64), BF16(1, 64),
                I64(1), BF16(1, 1, 288, 512),
                BF16(1, 1, 128, 64), F32(1, 6144),
                F32(1, 1536),
                rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5),
        # 支持mla MMCq+MMCkvKr+MMQcQr kvcache量化,token_x:B=1, s=1, He=7168; weight_dq:He=7168, Hcq=1536;
        # weight_uq_qr:Hcq=1536, N*(D+Dr)=(128+64)*128; weight_uk:N=128, D=128, Hckv=512;
        # weight_dkv_kr:He=7168, Hckv+Dr=512+64; rms_norm_weight:Hcq=1536;
        # rmsnorm_gamma_ckv:Hckv=512; rope_sin:B=1, S=1, Dr=64; rope_cos:B=1, S=1, Dr=64;
        # cache_index:B=1, S=1; kv_cache:B=1, Nkv=1, Skv=288, Hckv=512;
        # kr_cache:B=1, Nkv=1, Skv=288, Dr=64;dequant_scale_x:BS1=1, 1;
        # dequant_scale_w_dq:1, Hcq=1536; dequant_scale_w_uq_qr:1, N*(D+Dr)=(128+64)*128;
        # dequant_scale_w_dkv_kr:1, Hckv+Dr=512+64; quant_scale_ckv:1, Hckv=512;
        # quant_scale_ckr:1, Dr=64; smooth_scales_cq:1, Hcq=1536;
        Support(I8(1, 1, 7168), I8(7168, 1536),
                I8(1536, 6144), BF16(128, 128, 512),
                I8(7168, 576), BF16(1536),
                BF16(512), BF16(1, 1, 64), BF16(1, 1, 64),
                I64(1, 1), I8(1, 1, 288, 512),
                I8(1, 1, 288, 64), F32(1, 1),
                F32(1, 1536), F32(1, 6144),
                F32(1, 576), F32(1, 512),
                F32(1, 64), F32(1, 1536),
                rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5),
        # 支持mla MMCq+MMCkvKr+MMQcQr kvcache量化,BS合轴,token_x:T=1, He=7168; weight_dq:He=7168, Hcq=1536;
        # weight_uq_qr:Hcq=1536, N*(D+Dr)=(128+64)*128; weight_uk:N=128, D=128, Hckv=512;
        # weight_dkv_kr:He=7168, Hckv+Dr=512+64; rms_norm_weight:Hcq=1536;
        # rmsnorm_gamma_ckv:Hckv=512; rope_sin:T=1, Dr=64; rope_cos:T=1, Dr=64;
        # cache_index:T=1; kv_cache:B=1, Nkv=1, Skv=288, Hckv=512;
        # kr_cache:B=1, Nkv=1, Skv=288, Dr=64; dequant_scale_x:T=1, 1;
        # dequant_scale_w_dq:1, Hcq=1536; dequant_scale_w_uq_qr:1, N*(D+Dr)=(128+64)*128;
        # dequant_scale_w_dkv_kr:1, Hckv+Dr=512+64; quant_scale_ckv:1, Hckv=512;
        # quant_scale_ckr:1, Dr=64; smooth_scales_cq:1, Hcq=1536;
        Support(I8(1, 7168), I8(7168, 1536),
                I8(1536, 6144), BF16(128, 128, 512),
                I8(7168, 576), BF16(1536),
                BF16(512), BF16(1, 64), BF16(1, 64),
                I64(1), I8(1, 1, 288, 512),
                I8(1, 1, 128, 64), F32(1, 1),
                F32(1, 1536), F32(1, 6144),
                F32(1, 576), F32(1, 512),
                F32(1, 64), F32(1, 1536),
                rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5),
        # 支持mla MMCq+MMCkvKr+MMQcQr kvcache非量化,token_x:B=1, s=1, He=7168; weight_dq:He=7168, Hcq=1536;
        # weight_uq_qr:Hcq=1536, N*(D+Dr)=(128+64)*128; weight_uk:N=128, D=128, Hckv=512;
        # weight_dkv_kr:He=7168, Hckv+Dr=512+64; rms_norm_weight:Hcq=1536;
        # rmsnorm_gamma_ckv:Hckv=512; rope_sin:B=1, S=1, Dr=64; rope_cos:B=1, S=1, Dr=64;
        # cache_index:B=1, S=1; kv_cache:B=1, Nkv=1, Skv=288, Hckv=512;
        # kr_cache:B=1, Nkv=1, Skv=288, Dr=64;dequant_scale_x:BS1=1, 1;
        # dequant_scale_w_dq:1, Hcq=1536; dequant_scale_w_uq_qr:1, N*(D+Dr)=(128+64)*128;
        # dequant_scale_w_dkv_kr:1, Hckv+Dr=512+64; smooth_scales_cq:1, Hcq=1536;
        Support(I8(1, 1, 7168), I8(7168, 1536),
                I8(1536, 6144), BF16(128, 128, 512),
                I8(7168, 576), BF16(1536),
                BF16(512), BF16(1, 1, 64), BF16(1, 1, 64),
                I64(1, 1), BF16(1, 1, 288, 512),
                BF16(1, 1, 288, 64), F32(1, 1),
                F32(1, 1536), F32(1, 6144),
                F32(1, 576), F32(1, 1536),
                rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5),
        # 支持mla MMCq+MMCkvKr+MMQcQr kvcache非量化,BS合轴,token_x:T=1, He=7168; weight_dq:He=7168, Hcq=1536;
        # weight_uq_qr:Hcq=1536, N*(D+Dr)=(128+64)*128; weight_uk:N=128, D=128, Hckv=512;
        # weight_dkv_kr:He=7168, Hckv+Dr=512+64; rms_norm_weight:Hcq=1536;
        # rmsnorm_gamma_ckv:Hckv=512; rope_sin:T=1, Dr=64; rope_cos:T=1, Dr=64;
        # cache_index:T=1; kv_cache:B=1, Nkv=1, Skv=288, Hckv=512;
        # kr_cache:B=1, Nkv=1, Skv=288, Dr=64; dequant_scale_x:T=1, 1;
        # dequant_scale_w_dq:1, Hcq=1536; dequant_scale_w_uq_qr:1, N*(D+Dr)=(128+64)*128;
        # dequant_scale_w_dkv_kr:1, Hckv+Dr=512+64; smooth_scales_cq:1, Hcq=1536;
        Support(BF16(1, 7168), BF16(7168, 1536),
                I8(1536, 6144), BF16(128, 128, 512),
                BF16(7168, 576), BF16(1536),
                BF16(512), BF16(1, 64), BF16(1, 64),
                I64(1), BF16(1, 1, 288, 512),
                BF16(1, 1, 128, 64), F32(1, 1),
                F32(1, 1536), F32(1, 6144),
                F32(1, 576), F32(1, 1536),
                rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_mla_prolog.default)
def convert_npu_npu_mla_prolog(
    token_x: Tensor,
    weight_dq: Tensor,
    weight_uq_qr: Tensor,
    weight_uk: Tensor,
    weight_dkv_kr: Tensor,
    rmsnorm_gamma_cq: Tensor,
    rmsnorm_gamma_ckv: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
    cache_index: Tensor,
    kv_cache: Tensor,
    kr_cache: Tensor,
    *,
    dequant_scale_x: Optional[Tensor] = None,
    dequant_scale_w_dq: Optional[Tensor] = None,
    dequant_scale_w_uq_qr: Optional[Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[Tensor] = None,
    quant_scale_ckv: Optional[Tensor] = None,
    quant_scale_ckr: Optional[Tensor] = None,
    smooth_scales_cq: Optional[Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    meta_outputs: TensorSpec = None
):

    '''NB: npu_mla_prolog(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, Tensor cache_index, Tensor kv_cache, Tensor kr_cache, *, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str cache_mode="PA_BSND") -> (Tensor, Tensor, Tensor, Tensor)'''

    return ge.MlaProlog(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, dequant_scale_x=dequant_scale_x,
        dequant_scale_w_dq=dequant_scale_w_dq, dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr, quant_scale_ckv=quant_scale_ckv,
        quant_scale_ckr=quant_scale_ckr, smooth_scales_cq=smooth_scales_cq,
        rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)


@register_fx_node_ge_converter(torch.ops.npu.npu_mla_prolog_v2.default)
def convert_npu_npu_mla_prolog_v2(
    token_x: Tensor,
    weight_dq: Tensor,
    weight_uq_qr: Tensor,
    weight_uk: Tensor,
    weight_dkv_kr: Tensor,
    rmsnorm_gamma_cq: Tensor,
    rmsnorm_gamma_ckv: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
    cache_index: Tensor,
    kv_cache: Tensor,
    kr_cache: Tensor,
    *,
    dequant_scale_x: Optional[Tensor] = None,
    dequant_scale_w_dq: Optional[Tensor] = None,
    dequant_scale_w_uq_qr: Optional[Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[Tensor] = None,
    quant_scale_ckv: Optional[Tensor] = None,
    quant_scale_ckr: Optional[Tensor] = None,
    smooth_scales_cq: Optional[Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    meta_outputs: TensorSpec = None
):

    '''NB: npu_mla_prolog_v2(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, Tensor cache_index, Tensor kv_cache, Tensor kr_cache, *, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str cache_mode="PA_BSND") ->(Tensor, Tensor, Tensor, Tensor, Tensor)'''

    return ge.MlaPrologV2(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, dequant_scale_x=dequant_scale_x,
        dequant_scale_w_dq=dequant_scale_w_dq, dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr, quant_scale_ckv=quant_scale_ckv,
        quant_scale_ckr=quant_scale_ckr, smooth_scales_cq=smooth_scales_cq,
        rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
