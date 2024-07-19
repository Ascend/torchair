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
from torch._ops import OpOverload
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, get_checkpoint_func
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.higher_order.run_with_rng_state)
def converter_run_with_rng_state_default(
    rng_state: Tensor,
    op: OpOverload,
    *args,
    meta_outputs: TensorSpec = None,
    **kwargs
):
    fn = get_checkpoint_func(op)
    _, result = fn(*args, meta_outputs=meta_outputs, **kwargs, rng_state=rng_state)
    return result