from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._scaled_dot_product_efficient_attention.default)
def conveter_aten__scaled_dot_product_efficient_attention_default(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    compute_log_sumexp: bool,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_scaled_dot_product_efficient_attention(Tensor query, Tensor key, Tensor value, bool compute_log_sumexp, bool is_causal=False, *, float? scale=None) -> (Tensor output, Tensor log_sumexp)"""
    raise NotImplementedError(
        "torch.ops.aten._scaled_dot_product_efficient_attention.default ge_converter is not implemented!"
    )
