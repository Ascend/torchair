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
from torchair.ge import attr


@register_fx_node_ge_converter(torch.ops.npu.npu_kv_quant_sparse_flash_attention.default)
def convert_npu_kv_quant_sparse_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sparse_indices: Tensor,
    scale_value: float,
    key_quant_mode: int,
    value_quant_mode: int,
    *,
    key_dequant_scale: Optional[Tensor] = None,
    value_dequant_scale: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    actual_seq_lengths_query: Optional[Tensor] = None,
    actual_seq_lengths_kv: Optional[Tensor] = None,
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    attention_mode: int = 0,
    quant_scale_repo_mode: int = 1,
    tile_size: int = 128,
    rope_head_dim: int = 64,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "KvQuantSparseFlashAttention",
        inputs={"query": query, 
                "key": key,
                "value": value,
                "sparse_indices": sparse_indices,
                "key_dequant_scale": key_dequant_scale,
                "value_dequant_scale": value_dequant_scale,
                "block_table": block_table,
                "actual_seq_lengths_query": actual_seq_lengths_query,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
               },
        attrs={"scale_value": attr.Float(scale_value),
               "key_quant_mode": attr.Int(key_quant_mode),
               "value_quant_mode": attr.Int(value_quant_mode),
               "sparse_block_size": attr.Int(sparse_block_size),
               "layout_query": attr.Str(layout_query),
               "layout_kv": attr.Str(layout_kv),
               "sparse_mode": attr.Int(sparse_mode),
               "pre_tokens": attr.Int(pre_tokens),
               "next_tokens": attr.Int(next_tokens),
               "attention_mode": attr.Int(attention_mode),
               "quant_scale_repo_mode": attr.Int(quant_scale_repo_mode),
               "tile_size": attr.Int(tile_size),
               "rope_head_dim": attr.Int(rope_head_dim),
               },
        outputs=['attention_out']
    )