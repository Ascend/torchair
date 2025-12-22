from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_apply_rotary_pos_emb.default)
def conveter_npu_apply_rotary_pos_emb_default(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    layout: str = 'BSH',
    rotary_mode: str = 'half',
    meta_outputs: List[TensorSpec] = None,
):
    """
    NB: npu::npu_apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, str layout='BSH') 
                                      -> (Tensor, Tensor)
    """
    """
    Warning: kernel [npu_apply_rotary_pos_emb] is not a out-of-place op, but used as in-place op.
             This current usage may cause the input to be changed unexpectedly, and the caller 
             needs to pay attention to this feature.
    """
    support_layer = ['BSH', 'BSND', 'SBND', 'BNSD', 'TND']
    support_rotary_mode = ['half', 'quarter', 'interleave']
    if layout not in support_layer:
        raise NotImplementedError("layout only support BSH/BSND/SBND/BNSD/TND now!")
    if rotary_mode not in support_rotary_mode:
        raise NotImplementedError("rotary_mode only support half/quarter/interleave now!")

    layout_val = 1
    if layout == "SBND":
        layout_val = 2
    elif layout == "BNSD":
        layout_val = 3
    elif layout == "TND":
        layout_val = 4

    tm_q = ge.TensorMove(query)
    tm_k = ge.TensorMove(key)
    return ge.ApplyRotaryPosEmb(tm_q, tm_k, cos, sin, layout=layout_val, rotary_mode=rotary_mode)