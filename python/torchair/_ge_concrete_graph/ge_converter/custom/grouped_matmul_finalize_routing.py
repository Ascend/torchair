from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
])
@register_fx_node_ge_converter(torch.ops.npu.npu_grouped_matmul_finalize_routing.default)
def conveter_npu_grouped_matmul_finalize_routing(
    x: Tensor,
    w: Tensor,
    group_list: Tensor,
    *,
    scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
    pertoken_scale: Optional[Tensor] = None,
    shared_input: Optional[Tensor] = None, 
    logit: Optional[Tensor] = None,
    row_index: Optional[Tensor] = None,
    dtype: torch.dtype = None,
    shared_input_weight: float = 1.0,
    shared_input_offset: int = 0,
    output_bs: int = 0,
    group_list_type: int = 1,
    tuning_config: Optional[List[int]] = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_grouped_matmul_finalize_routing(Tensor x, Tensor w, Tensor group_list, *,
                        Tensor? scale=None, Tensor? bias=None, Tensor? offset=None,
                        Tensor? pertoken_scale=None, Tensor? shared_input=None,
                        Tensor? logit=None, Tensor? row_index=None, ScalarType? dtype=None,
                        float? shared_input_weight=1.0, int? shared_input_offset=0,
                        int? output_bs=0, int? group_list_type=1, int[] tuning_config=[]) -> Tensor
    """
    tuning_config = tuning_config or [0]
    if dtype is None or dtype == torch.float32:
        dtype = DataType.DT_FLOAT
    else:
        raise RuntimeError("Not supported output dtype is " + str(dtype))
    
    opp_ver = get_cann_opp_version()
    second_dot_index = opp_ver.find('.', opp_ver.find('.') + 1)
    opp_ver = float(opp_ver[:second_dot_index])
    if w.dtype == DataType.DT_INT32:
        if opp_ver >= 8.5:
            new_w = ge.Bitcast(w, type=DataType.DT_INT4, keep_dim=True)
        else:
            const_w = ge.Const([1] * (w.rank - 1) + [8])
            shape_w = ge.Shape(w)
            shape_w = ge.Mul(shape_w, const_w)
            new_w = ge.Bitcast(w, type=DataType.DT_INT4)
            new_w = ge.Reshape(new_w, shape_w)
    else:
        new_w = w

    if output_bs == 0:
        output_bs = x.symsize[0]
    return ge.GroupedMatmulFinalizeRouting(x,
                                 new_w,
                                 scale=scale,
                                 bias=bias,
                                 pertoken_scale=pertoken_scale,
                                 group_list=group_list,
                                 shared_input=shared_input,
                                 logit=logit,
                                 row_index=row_index,
                                 offset=offset,
                                 dtype=dtype,
                                 shared_input_weight=shared_input_weight,
                                 shared_input_offset=shared_input_offset,
                                 transpose_x=False,
                                 transpose_w=False,
                                 output_bs=output_bs,
                                 group_list_type=group_list_type,
                                 tuning_config=tuning_config)
