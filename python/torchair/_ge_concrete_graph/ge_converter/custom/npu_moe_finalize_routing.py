from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import numpy as np
import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F16, F32, T, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F16(24, 7), F16(12, 7), F16(12, 7), F16(3, 7), F16(12, 2),
            T(value=np.arange(24).astype(np.int32), dtype=torch.int32),
            T(value=np.random.randint(low=0, high=3, size=(12, 2)).astype(np.int32), dtype=torch.int32)),
    Support(F16(24, 7), F16(12, 7), None, F16(3, 7), F16(12, 2),
            T(value=np.arange(24).astype(np.int32), dtype=torch.int32),
            T(value=np.random.randint(low=0, high=3, size=(12, 2)).astype(np.int32), dtype=torch.int32)),
    Support(F32(24, 7), F32(12, 7), None, F32(3, 7), F32(12, 2),
            T(value=np.arange(24).astype(np.int32), dtype=torch.int32),
            T(value=np.random.randint(low=0, high=3, size=(12, 2)).astype(np.int32), dtype=torch.int32)),
    Support(F32(24, 7), None, None, F32(3, 7), F32(12, 2),
            T(value=np.arange(24).astype(np.int32), dtype=torch.int32),
            T(value=np.random.randint(low=0, high=3, size=(12, 2)).astype(np.int32), dtype=torch.int32)),
    Support(F32(24, 7), None, None, None, F32(12, 2),
            T(value=np.arange(24).astype(np.int32), dtype=torch.int32),
            T(value=np.random.randint(low=0, high=3, size=(12, 2)).astype(np.int32), dtype=torch.int32)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_moe_finalize_routing.default)
def conveter_npu_moe_finalize_routing_default(
    expanded_permuted_rows: Tensor,
    skip1: Optional[Tensor],
    skip2_optional: Optional[Tensor],
    bias: Optional[Tensor],
    scales: Optional[Tensor],
    expanded_src_to_dst_row: Tensor,
    expert_for_source_row: Optional[Tensor],
    drop_pad_mode: Optional[int] = 0,
    meta_outputs: TensorSpec = None
):
    '''NB: npu::npu_moe_finalize_routing(Tensor expanded_permuted_rows, Tensor? skip1, Tensor? skip2, Tensor? bias, 
    Tensor? scales, Tensor expanded_src_to_dst_row, Tensor? export_for_source_row, int? drop_pad_mode=0) -> Tensor'''
    if skip1 is not None and bias is not None and scales is not None and \
                 expert_for_source_row is not None and drop_pad_mode == 0:
        return ge.MoeFinalizeRouting(expanded_permuted_rows, skip1, skip2_optional, bias, scales, 
                                     expanded_src_to_dst_row, expert_for_source_row)
    else:
        return ge.MoeFinalizeRoutingV2(expanded_permuted_rows, expanded_src_to_dst_row, skip1, skip2_optional, bias,
                                       scales, expert_for_source_row, drop_pad_mode=drop_pad_mode)
