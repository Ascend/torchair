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
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(16, 128, 768), I64(16, 128), 3, 0, False),
        Support(F32(16, 128, 768), I64(16, 128), 20006, 0, False),
        Support(F32(16, 128, 768), I64(16, 128), 3, 0, True),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.embedding_dense_backward.default)
def conveter_aten_embedding_dense_backward_default(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: Union[int, Tensor],
    padding_idx: Union[int, Tensor],
    scale_grad_by_freq: bool,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor"""
    if isinstance(num_weights, Tensor) or isinstance(padding_idx, Tensor):
        raise RuntimeError("ge.EmbeddingDenseGrad is not support when num_weights or padding_idx is Tensor")
    if scale_grad_by_freq:
        indices = ge.Cast(indices, dst_type=DataType.DT_INT32)
    return ge.EmbeddingDenseGrad(grad_output, indices,
                                    num_weights=num_weights,
                                    padding_idx=padding_idx,
                                    scale_grad_by_freq=scale_grad_by_freq)


@register_fx_node_ge_converter(torch.ops.aten.embedding_dense_backward.out)
def conveter_aten_embedding_dense_backward_out(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: Union[int, Tensor],
    padding_idx: Union[int, Tensor],
    scale_grad_by_freq: bool,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::embedding_dense_backward.out(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.embedding_dense_backward.out ge_converter is not implemented!")
