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
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, BF16, F32, F16, F64, I32, I16, \
    I64, I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote, DataType

    
@declare_supported([
    Support(BF16(4, 16, 32), BF16(4, 16, 32), BF16(4, 16, 32), head_num=32, 
            input_layout='BSH', pse=None, padding_mask=None, atten_mask=BOOL(4, 1, 16, 16), 
            scale_value=0.0883, keep_prob=1.0, pre_tockens=4096, next_tockens=0),
    Support(BF16(4, 32, 2048, 128), BF16(4, 32, 2048, 128), BF16(4, 32, 2048, 128), head_num=32, 
            input_layout='BNSD', pse=None, padding_mask=None, atten_mask=BOOL(4, 1, 2048, 2048), 
            scale_value=0.0883, keep_prob=1.0, pre_tockens=4096, next_tockens=0),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_fusion_attention.default)
def conveter_npu_fusion_attention_default(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    head_num: int,
    input_layout: str,
    pse: Optional[Tensor] = None,
    padding_mask: Optional[Tensor] = None,
    atten_mask: Optional[Tensor] = None,
    scale: float = 1.0,
    keep_prob: float = 1.0,
    pre_tockens: int = 2147483647,
    next_tockens: int = 2147483647, 
    inner_precise: int = 0,
    prefix: Optional[List[int]] = None, 
    actual_seq_qlen: Optional[List[int]] = None, 
    actual_seq_kvlen: Optional[List[int]] = None, 
    sparse_mode: int = 0, 
    gen_mask_parallel: bool = True, 
    sync: bool = False,
    meta_outputs: TensorSpec = None
):
    """
    NB: npu::npu_fusion_attention(Tensor query, Tensor key, Tensor value, int head_num, str input_layout, 
    Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, float scale=1., float keep_prob=1., 
    int pre_tockens=2147483647, int next_tockens=2147483647, int inner_precise=0, int[]? prefix=None, 
    int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, int sparse_mode=0, 
    bool gen_mask_parallel=True, bool sync=False) -> (Tensor, Tensor, Tensor, Tensor, int, int, int)
    """
    is_bsh_available = True if input_layout == "BSH" and keep_prob == 1.0 else False
    is_bnsd_available = True if input_layout == "BNSD" and keep_prob == 1.0 else False
    q_start_idx = 0
    kv_start_idx = 0

    if is_bsh_available or is_bnsd_available:  
        # if keep_prob == 1.0, no dropout will be performed, so the seed, offset, and numels will have no effect.
        seed, offset, numels = 0, 0, 0
        ret = ge.FlashAttentionScore(
            query,
            key,
            value,
            real_shift=None,
            drop_mask=None,
            padding_mask=padding_mask,
            atten_mask=atten_mask,
            prefix=prefix,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            q_start_idx=q_start_idx,
            kv_start_idx=kv_start_idx,
            head_num=head_num,
            input_layout=input_layout,
            scale_value=scale,
            keep_prob=keep_prob,
            pre_tockens=pre_tockens,
            next_tockens=next_tockens,
            inner_precise=inner_precise,
            sparse_mode=sparse_mode
        )
        softmax_max, softmax_sum, softmax_out, attention_out = ret
        corr_ret = (attention_out, softmax_max, softmax_sum, softmax_out)
        dropout_ret = (seed, offset, numels)
        return *corr_ret, *dropout_ret
    else:
        raise NotImplementedError(
                f"torch.ops.npu.npu_fusion_attention.default ge_converter "
                f"is not implemented when input_layout != BSH & BNSD or keep_prob != 1.0 !")
    
    
@declare_supported([
    Support(BF16(4, 16, 32), BF16(4, 16, 32), BF16(4, 16, 32), BF16(4, 16, 32), head_num=32, 
            input_layout='BSH', pse=None, padding_mask=None, atten_mask=BOOL(4, 1, 16, 16), 
            softmax_max=F32(4, 32, 16, 8), softmax_sum=F32(4, 32, 16, 8), softmax_in=BF16(0), 
            attention_in=BF16(4, 16, 32), scale_value=0.0883, keep_prob=1.0, pre_tockens=4096, next_tockens=0),
    Support(BF16(4, 32, 2048, 128), BF16(4, 32, 2048, 128), BF16(4, 32, 2048, 128), BF16(4, 32, 2048, 128), head_num=32, 
            input_layout='BNSD', pse=None, padding_mask=None, atten_mask=BOOL(4, 1, 2048, 2048), 
            softmax_max=F32(4, 32, 2048, 8), softmax_sum=F32(4, 32, 2048, 8), softmax_in=BF16(0), 
            attention_in=BF16(4, 32, 2048, 128), scale_value=0.0883, keep_prob=1.0, pre_tockens=4096, next_tockens=0),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_fusion_attention_grad.default)
def conveter_npu_fusion_attention_grad_default(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dy: Tensor,
    head_num: int,
    input_layout: str,
    pse: Optional[Tensor] = None,
    padding_mask: Optional[Tensor] = None,
    atten_mask: Optional[Tensor] = None,
    softmax_max: Optional[Tensor] = None,
    softmax_sum: Optional[Tensor] = None,
    softmax_in: Optional[Tensor] = None,
    attention_in: Optional[Tensor] = None,
    scale_value: float = 1.0,
    keep_prob: float = 1.0,
    pre_tockens: int = 2147483647,
    next_tockens: int = 2147483647, 
    inner_precise: int = 0,
    seed: int = 0,
    offset: int = 0,
    numels: int = 0,
    prefix: Optional[List[int]] = None, 
    actual_seq_qlen: Optional[List[int]] = None, 
    actual_seq_kvlen: Optional[List[int]] = None, 
    sparse_mode: int = 0, 
    gen_mask_parallel: bool = True, 
    sync: bool = False,
    meta_outputs: TensorSpec = None
):
    """
    NB: npu:npu_fusion_attention_grad(Tensor query, Tensor key, Tensor value, Tensor dy, int head_num, 
    str input_layout, *, Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? 
    softmax_max=None, Tensor? softmax_sum=None, Tensor? softmax_in=None, Tensor? attention_in=None, 
    float scale_value=1., float keep_prob=1., int pre_tockens=2147483647, int next_tockens=2147483647, 
    int inner_precise=0, int seed=0, int offset=0, int numels=0, int[]? prefix=None, int[]? actual_seq_qlen=None, 
    int[]? actual_seq_kvlen=None, int sparse_mode=0, bool gen_mask_parallel=True,  bool sync=False) 
    -> (Tensor, Tensor, Tensor, Tensor)
    """
    is_bsh_available = True if input_layout == "BSH" and keep_prob == 1.0 else False
    is_bnsd_available = True if input_layout == "BNSD" and keep_prob == 1.0 else False
    q_start_idx = 0
    kv_start_idx = 0

    if is_bsh_available or is_bnsd_available:  
        # if keep_prob == 1.0, no dropout will be performed, so the seed, offset, and numels will have no effect.
        ret = ge.FlashAttentionScoreGrad(
            query,
            key,
            value,
            dy,
            pse_shift=None,
            drop_mask=None,
            padding_mask=padding_mask,
            atten_mask=atten_mask,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            softmax_in=softmax_in,
            attention_in=attention_in,
            prefix=prefix,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            q_start_idx=q_start_idx,
            kv_start_idx=kv_start_idx,
            head_num=head_num,
            input_layout=input_layout,
            scale_value=scale_value,
            keep_prob=keep_prob,
            pre_tockens=pre_tockens,
            next_tockens=next_tockens,
            inner_precise=inner_precise,
            sparse_mode=sparse_mode
        )
        
        return ret
    else:
        raise NotImplementedError(
                f"torch.ops.npu.npu_fusion_attention_grad.default ge_converter "
                f"is not implemented when input_layout != BSH & BNSD or keep_prob != 1.0 !")