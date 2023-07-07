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


@register_fx_node_ge_converter(torch.ops.aten._embedding_bag_forward_only.default)
def conveter_aten__embedding_bag_forward_only_default(
        weight: Tensor,
        indices: Tensor,
        offsets: Tensor,
        scale_grad_by_freq: bool = False,
        mode: int = 0,
        sparse: bool = False,
        per_sample_weights: Optional[Tensor] = None,
        include_last_offset: bool = False,
        padding_idx: int = -1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten._embedding_bag_forward_only.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._embedding_bag_forward_only.out)
def conveter_aten__embedding_bag_forward_only_out(
        weight: Tensor,
        indices: Tensor,
        offsets: Tensor,
        scale_grad_by_freq: bool = False,
        mode: int = 0,
        sparse: bool = False,
        per_sample_weights: Optional[Tensor] = None,
        include_last_offset: bool = False,
        padding_idx: int = -1,
        *,
        out0: Tensor = None,
        out1: Tensor = None,
        out2: Tensor = None,
        out3: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_embedding_bag_forward_only.out(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!)) """
    raise NotImplementedError("torch.ops.aten._embedding_bag_forward_only.out ge converter is not implement!")


