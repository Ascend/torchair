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
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support


@register_fx_node_ge_converter(torch.ops.npu.npu_alltoallv_gmm.default)
def convert_npu_alltoallv_gmm(
    gmm_x: Tensor,
    gmm_weight: Tensor,
    hcom: str,
    ep_world_size: int,
    send_counts: List[int],
    recv_counts: List[int],
    *,
    send_counts_tensor: Optional[Tensor] = None,
    recv_counts_tensor: Optional[Tensor] = None,
    mm_x: Optional[Tensor] = None,
    mm_weight: Optional[Tensor] = None,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
    permute_out_flag: bool = False,
    meta_outputs: TensorSpec = None
):
    return ge.AlltoAllvGroupedMatMul(
        gmm_x=gmm_x,
        gmm_weight=gmm_weight,
        send_counts_tensor=send_counts_tensor,
        recv_counts_tensor=recv_counts_tensor,
        mm_x=mm_x,
        mm_weight=mm_weight,
        group=hcom,
        ep_world_size=ep_world_size,
        send_counts=send_counts,
        recv_counts=recv_counts,
        trans_gmm_weight=trans_gmm_weight,
        trans_mm_weight=trans_mm_weight,
        permute_out_flag=permute_out_flag)