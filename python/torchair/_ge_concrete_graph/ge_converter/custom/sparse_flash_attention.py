from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
import torchair
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr


@register_fx_node_ge_converter(torch.ops.npu.npu_sparse_flash_attention.default)
def convert_npu_sparse_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sparse_indices: Tensor,
    scale_value: float,
    *,
    block_table: Optional[Tensor] = None,
    actual_seq_lengths_query: Optional[Tensor] = None,
    actual_seq_lengths_kv: Optional[Tensor] = None,
    query_rope: Optional[Tensor] = None,
    key_rope: Optional[Tensor] = None,
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = (1 << 63) - 1,
    next_tokens: int = (1 << 63) - 1,
    attention_mode: int = 0,
    return_softmax_lse: bool = False,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "SparseFlashAttention",
        inputs={
            "query": query,
            "key": key,
            "value": value,
            "sparse_indices": sparse_indices,
            "block_table": block_table,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "query_rope": query_rope,
            "key_rope": key_rope,
        },
        attrs={
            "scale_value": attr.Float(scale_value),
            "sparse_block_size": attr.Int(sparse_block_size),
            "layout_query": attr.Str(layout_query),
            "layout_kv": attr.Str(layout_kv),
            "sparse_mode": attr.Int(sparse_mode),
            "pre_tokens": attr.Int(pre_tokens),
            "next_tokens": attr.Int(next_tokens),
            "attention_mode": attr.Int(attention_mode),
            "return_softmax_lse": attr.Bool(return_softmax_lse),
        },
        outputs=[
            "attention_out",
            "softmax_max",
            "softmax_sum",
        ]
    )