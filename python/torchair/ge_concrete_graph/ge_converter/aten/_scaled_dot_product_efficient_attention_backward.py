import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
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


@register_fx_node_ge_converter(torch.ops.aten._scaled_dot_product_efficient_attention_backward.default)
def conveter_aten__scaled_dot_product_efficient_attention_backward_default(
        grad_out_: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        out: Tensor,
        logsumexp: Tensor,
        is_causal: bool = False,
        chunk_grad_outputs: bool = False,
        *,
        scale: Optional[float] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_scaled_dot_product_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, bool is_causal=False, bool chunk_grad_outputs=False, *, float? scale=None) -> (Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten._scaled_dot_product_efficient_attention_backward.default ge converter is not implement!")


