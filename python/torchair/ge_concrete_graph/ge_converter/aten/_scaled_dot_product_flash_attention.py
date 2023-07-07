import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten._scaled_dot_product_flash_attention.default)
def conveter_aten__scaled_dot_product_flash_attention_default(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        return_debug_mask: bool = False,
        *,
        scale: Optional[float] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor ouput, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, int max_q, int max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask) """
    raise NotImplementedError("torch.ops.aten._scaled_dot_product_flash_attention.default ge converter is not implement!")


