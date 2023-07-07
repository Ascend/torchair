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


@register_fx_node_ge_converter(torch.ops.prims.scalar_tensor.default)
def conveter_prims_scalar_tensor_default(
        s: Union[Number, Tensor],
        *,
        dtype: Optional[int] = None,
        device: Optional[Device] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: prims::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Device? device=None) -> Tensor """
    raise NotImplementedError("torch.ops.prims.scalar_tensor.default ge converter is not implement!")


