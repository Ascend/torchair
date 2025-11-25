from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge._ge_graph import DataType, torch_dtype_value_to_ge_proto_type, torch_dtype_value_to_ge_type
from torchair._utils.check_platform import is_arch35
from torchair.ge._ge_graph import dont_prune_me


@register_fx_node_ge_converter(torch.ops.npu.npu_add_quant_gmm.default)
def conveter_npu_npu_add_quant_gmm(
    y: Tensor,
    x1: Tensor,
    x2: Tensor,
    x2_scale: Tensor,
    group_list: Tensor,
    *,
    x1_scale: Optional[Tensor] = None,
    group_list_type: Optional[int] = 0,
    group_sizes: Optional[List[int]] = None,
    x1_dtype: Optional[int] = None,
    x2_dtype: Optional[int] = None,
    x1_scale_dtype: Optional[int] = None,
    x2_scale_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """
    kernel [npu_add_quant_gmm] is a out-of-place op, but it is supported by corresponding in-place op
    npu::npu_add_quant_gmm(Tensor(a!) self, Tensor x1, Tensor x2, Tensor x2_scale, Tensor group_list, *,
                            Tensor? x1_scale=None, int? group_list_type=0, int[]? group_sizes=None,
                            int? x1_dtype=None, int? x2_dtype=None, int? x1_scale_dtype=None,
                            int? x2_scale_dtype=None) -> Tensor(a!)
    """
    if not is_arch35():
        raise RuntimeError("This operator currently only supports Ascend910_95 chips.")
    group_size = 0
    if x1_dtype is not None:
        x1 = ge.Bitcast(x1, type=torch_dtype_value_to_ge_type(x1_dtype))
        x1.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_dtype)
    if x2_dtype is not None:
        x2 = ge.Bitcast(x2, type=torch_dtype_value_to_ge_type(x2_dtype))
        x2.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_dtype)
    if x1_scale is not None and x1_scale_dtype is not None:
        x1_scale = ge.Bitcast(x1_scale, type=torch_dtype_value_to_ge_type(x1_scale_dtype))
        x1_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_scale_dtype)
    if x2_scale_dtype is not None:
        x2_scale = ge.Bitcast(x2_scale, type=torch_dtype_value_to_ge_type(x2_scale_dtype))
        x2_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_scale_dtype)

    y_copy = ge.TensorMove(y)

    out = ge.QuantGroupedMatmulInplaceAdd(x1,
                                            x2,
                                            scale2=x2_scale,
                                            group_list=group_list,
                                            y=y_copy,
                                            scale1=x1_scale,
                                            group_list_type=group_list_type,
                                            group_size=group_size)

    return out


@register_fx_node_ge_converter(torch.ops.npu.npu_add_quant_gmm_.default)
def conveter_npu_npu_add_quant_gmm(
    y: Tensor,
    x1: Tensor,
    x2: Tensor,
    x2_scale: Tensor,
    group_list: Tensor,
    *,
    x1_scale: Optional[Tensor] = None,
    group_list_type: Optional[int] = 0,
    group_sizes: Optional[List[int]] = None,
    x1_dtype: Optional[int] = None,
    x2_dtype: Optional[int] = None,
    x1_scale_dtype: Optional[int] = None,
    x2_scale_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """
    kernel [npu_add_quant_gmm] is an in-place op
    npu::npu_add_quant_gmm_(Tensor(a!) self, Tensor x1, Tensor x2, Tensor x2_scale, Tensor group_list, *,
                            Tensor? x1_scale=None, int? group_list_type=0, int[]? group_sizes=None,
                            int? x1_dtype=None, int? x2_dtype=None, int? x1_scale_dtype=None,
                            int? x2_scale_dtype=None) -> Tensor(a!)
    """
    if not is_arch35():
        raise RuntimeError("This operator currently only supports Ascend910_95 chips.")
    group_size = 0
    if x1_dtype is not None:
        x1 = ge.Bitcast(x1, type=torch_dtype_value_to_ge_type(x1_dtype))
        x1.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_dtype)
    if x2_dtype is not None:
        x2 = ge.Bitcast(x2, type=torch_dtype_value_to_ge_type(x2_dtype))
        x2.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_dtype)
    if x1_scale is not None and x1_scale_dtype is not None:
        x1_scale = ge.Bitcast(x1_scale, type=torch_dtype_value_to_ge_type(x1_scale_dtype))
        x1_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_scale_dtype)
    if x2_scale_dtype is not None:
        x2_scale = ge.Bitcast(x2_scale, type=torch_dtype_value_to_ge_type(x2_scale_dtype))
        x2_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_scale_dtype)

    out = ge.QuantGroupedMatmulInplaceAdd(x1,
                                            x2,
                                            scale2=x2_scale,
                                            group_list=group_list,
                                            y=y,
                                            scale1=x1_scale,
                                            group_list_type=group_list_type,
                                            group_size=group_size)
    
    dont_prune_me(out)
    return out
