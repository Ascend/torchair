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


def check_and_set_group_size(group_sizes):
    group_max = 65535 # group_m, group_n, group_k各占16位，组成64位group_siz, 因此每个值不能超过65535(16位的最大值)
    if(len(group_sizes) != 3):
        raise RuntimeError("group_size must be a list with 3 elements, actual group_sizes is " + str(group_sizes))
    group_m = group_sizes[0]
    group_n = group_sizes[1]
    group_k = group_sizes[2]
    if (group_m > group_max or group_n > group_max or group_k > group_max):
        raise RuntimeError("group_size cannot be larger than 65535, actual group_sizes is " + str(group_sizes))
    if (group_m < 0 or group_n < 0 or group_k < 0):
        raise RuntimeError("group_size cannot be smaller than 0, actual group_sizes is " + str(group_sizes))
    group_size = (group_m << 32) + (group_n << 16) + group_k
    return group_size


@register_fx_node_ge_converter(torch.ops.npu.npu_add_quant_matmul.default)
def conveter_npu_npu_add_quant_matmul(
    y: Tensor,
    x1: Tensor,
    x2: Tensor,
    x2_scale: Tensor,
    *,
    x1_scale: Optional[Tensor] = None,
    group_sizes: Optional[List[int]] = None,
    x1_dtype: Optional[int] = None,
    x2_dtype: Optional[int] = None,
    x1_scale_dtype: Optional[int] = None,
    x2_scale_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """
    kernel [npu_add_quant_matmul] is a out-of-place op, but it is supported by corresponding in-place op
    npu::npu_add_quant_matmul(Tensor(a!) self, Tensor x1, Tensor x2, Tensor x2_scale, *,
                            Tensor? x1_scale=None, int[]? group_sizes=None,
                            int? x1_dtype=None, int? x2_dtype=None, int? x1_scale_dtype=None,
                            int? x2_scale_dtype=None) -> Tensor(a!)
    """
    if not is_arch35():
        raise RuntimeError("This operator currently only supports Ascend910_95 chips.")
    group_size = 0
    if group_sizes is not None and isinstance(group_sizes, List):
        group_size = check_and_set_group_size(group_sizes)
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

    out = ge.QuantBatchMatmulInplaceAdd(x1,
                                        x2,
                                        x2_scale=x2_scale,
                                        y=y,
                                        x1_scale=x1_scale,
                                        transpose_x1=False,
                                        transpose_x2=False,
                                        group_size=group_size)

    return out


@register_fx_node_ge_converter(torch.ops.npu.npu_add_quant_matmul_.default)
def conveter_npu_npu_add_quant_matmul(
    y: Tensor,
    x1: Tensor,
    x2: Tensor,
    x2_scale: Tensor,
    *,
    x1_scale: Optional[Tensor] = None,
    group_sizes: Optional[List[int]] = None,
    x1_dtype: Optional[int] = None,
    x2_dtype: Optional[int] = None,
    x1_scale_dtype: Optional[int] = None,
    x2_scale_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """
    kernel [npu_add_quant_matmul] is an in-place op
    npu::npu_add_quant_matmul_(Tensor(a!) self, Tensor x1, Tensor x2, Tensor x2_scale, *,
                            Tensor? x1_scale=None, int[]? group_sizes=None,
                            int? x1_dtype=None, int? x2_dtype=None, int? x1_scale_dtype=None,
                            int? x2_scale_dtype=None) -> Tensor(a!)
    """
    if not is_arch35():
        raise RuntimeError("This operator currently only supports on Ascend 950PR/Ascend 950DT.")

    group_size = 0
    if group_sizes is not None and isinstance(group_sizes, List):
        group_size = check_and_set_group_size(group_sizes)
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

    out = ge.QuantBatchMatmulInplaceAdd(x1,
                                        x2,
                                        x2_scale=x2_scale,
                                        y=y,
                                        x1_scale=x1_scale,
                                        transpose_x1=False,
                                        transpose_x2=False,
                                        group_size=group_size)
    
    dont_prune_me(out)
    return out
