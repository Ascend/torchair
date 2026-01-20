from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import torch_dtype_value_to_ge_type


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

    if atten_mask is not None:
        if atten_mask.dtype == DataType.DT_FLOAT16:
            atten_mask = dtype_promote(atten_mask, target_dtype=DataType.DT_UINT8)

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


@register_fx_node_ge_converter(torch.ops.npu.npu_mla_prolog_v3.default)
def convert_npu_npu_mla_prolog_v3(
    token_x: Tensor,
    weight_dq: Tensor,
    weight_uq_qr: Tensor,
    weight_uk: Tensor,
    weight_dkv_kr: Tensor,
    rmsnorm_gamma_cq: Tensor,
    rmsnorm_gamma_ckv: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
    kv_cache: Tensor,
    kr_cache: Tensor,
    *,
    cache_index: Optional[Tensor] = None,
    dequant_scale_x: Optional[Tensor] = None,
    dequant_scale_w_dq: Optional[Tensor] = None,
    dequant_scale_w_uq_qr: Optional[Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[Tensor] = None,
    quant_scale_ckv: Optional[Tensor] = None,
    quant_scale_ckr: Optional[Tensor] = None,
    smooth_scales_cq: Optional[Tensor] = None,
    actual_seq_len: Optional[Tensor] = None,
    k_nope_clip_alpha: Optional[Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    meta_outputs: TensorSpec = None,
):
    """NB: npu_mla_prolog_v3(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, Tensor(a!) kv_cache, Tensor(b!) kr_cache, *, Tensor cache_index, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, Tensor? actual_seq_len=None, Tensor? k_nope_clip_alpha=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str cache_mode="PA_BSND", bool query_norm_flag=false, int weight_quant_mode=0, int kv_quant_mode=0, int query_quant_mode=0, int ckvkr_repo_mode=0, int quant_scale_repo_mode=0, int tile_size=128, float qc_qr_scale=1.0, float kc_scale=1.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"""
    # mxfp8全量化场景 输入int8数据类型伪装成float8_e8m0
    if dequant_scale_x is not None and weight_quant_mode == 3:
        dequant_scale_x = ge.Bitcast(dequant_scale_x, type=DataType.DT_FLOAT8_E8M0)
    if dequant_scale_w_dq is not None and weight_quant_mode == 3:
        dequant_scale_w_dq = ge.Bitcast(dequant_scale_w_dq, type=DataType.DT_FLOAT8_E8M0)
    if dequant_scale_w_uq_qr is not None and weight_quant_mode == 3:
        dequant_scale_w_uq_qr = ge.Bitcast(dequant_scale_w_uq_qr, type=DataType.DT_FLOAT8_E8M0)
    if dequant_scale_w_dkv_kr is not None and weight_quant_mode == 3:
        dequant_scale_w_dkv_kr = ge.Bitcast(dequant_scale_w_dkv_kr, type=DataType.DT_FLOAT8_E8M0)
    return ge.MlaPrologV3(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        kv_cache,
        kr_cache,
        cache_index=cache_index,
        dequant_scale_x=dequant_scale_x,
        dequant_scale_w_dq=dequant_scale_w_dq,
        dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
        quant_scale_ckv=quant_scale_ckv,
        quant_scale_ckr=quant_scale_ckr,
        smooth_scales_cq=smooth_scales_cq,
        actual_seq_len=actual_seq_len,
        k_nope_clip_alpha=k_nope_clip_alpha,
        rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
        cache_mode=cache_mode,
        query_norm_flag=query_norm_flag,
        weight_quant_mode=weight_quant_mode,
        kv_cache_quant_mode=kv_cache_quant_mode,
        query_quant_mode=query_quant_mode,
        ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode,
        tile_size=tile_size,
        qc_qr_scale=qc_qr_scale,
        kc_scale=kc_scale
    ) 


@register_fx_node_ge_converter(torch.ops.npu.npu_mla_prolog_v3_functional.default)
def conveter_npu_mla_prolog_v3_functional(
    token_x: Tensor,
    weight_dq: Tensor,
    weight_uq_qr: Tensor,
    weight_uk: Tensor,
    weight_dkv_kr: Tensor,
    rmsnorm_gamma_cq: Tensor,
    rmsnorm_gamma_ckv: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
    kv_cache: Tensor,
    kr_cache: Tensor,
    *,
    cache_index: Optional[Tensor] = None,
    dequant_scale_x: Optional[Tensor] = None,
    dequant_scale_w_dq: Optional[Tensor] = None,
    dequant_scale_w_uq_qr: Optional[Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[Tensor] = None,
    quant_scale_ckv: Optional[Tensor] = None,
    quant_scale_ckr: Optional[Tensor] = None,
    smooth_scales_cq: Optional[Tensor] = None,
    actual_seq_len: Optional[Tensor] = None,
    k_nope_clip_alpha: Optional[Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    meta_outputs: List[TensorSpec] = None,
):
    # mxfp8全量化场景 输入int8数据类型伪装成float8_e8m0
    if dequant_scale_x is not None and weight_quant_mode == 3:
        dequant_scale_x = ge.Bitcast(dequant_scale_x, type=DataType.DT_FLOAT8_E8M0)
    if dequant_scale_w_dq is not None and weight_quant_mode == 3:
        dequant_scale_w_dq = ge.Bitcast(dequant_scale_w_dq, type=DataType.DT_FLOAT8_E8M0)
    if dequant_scale_w_uq_qr is not None and weight_quant_mode == 3:
        dequant_scale_w_uq_qr = ge.Bitcast(dequant_scale_w_uq_qr, type=DataType.DT_FLOAT8_E8M0)
    if dequant_scale_w_dkv_kr is not None and weight_quant_mode == 3:
        dequant_scale_w_dkv_kr = ge.Bitcast(dequant_scale_w_dkv_kr, type=DataType.DT_FLOAT8_E8M0)
    # 保证非原地算子的输入不会被修改
    kv_cache_copy = ge.TensorMove(kv_cache)
    kr_cache_copy = ge.TensorMove(kr_cache)
    (
        query,
        query_rope,
        kv_cache_out,
        kr_cache_out,
        dequant_scale_q_nope,
        query_norm,
        dequant_scale_q_norm,
    ) = ge.MlaPrologV3(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        kv_cache_copy,
        kr_cache_copy,
        cache_index=cache_index,
        dequant_scale_x=dequant_scale_x,
        dequant_scale_w_dq=dequant_scale_w_dq,
        dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
        quant_scale_ckv=quant_scale_ckv,
        quant_scale_ckr=quant_scale_ckr,
        smooth_scales_cq=smooth_scales_cq,
        actual_seq_len=actual_seq_len,
        k_nope_clip_alpha=k_nope_clip_alpha,
        rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
        cache_mode=cache_mode,
        query_norm_flag=query_norm_flag,
        weight_quant_mode=weight_quant_mode,
        kv_cache_quant_mode=kv_cache_quant_mode,
        query_quant_mode=query_quant_mode,
        ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode,
        tile_size=tile_size,
        qc_qr_scale=qc_qr_scale,
        kc_scale=kc_scale
    )
    return (
        query,
        query_rope,
        dequant_scale_q_nope,
        query_norm,
        dequant_scale_q_norm,
        kv_cache_out,
        kr_cache_out,
    )