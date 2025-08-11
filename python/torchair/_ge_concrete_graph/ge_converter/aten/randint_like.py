from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.randint_like.default)
def conveter_aten_randint_like_default(
    self: Tensor,
    high: Union[int, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.randint_like.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randint_like.low_dtype)
def conveter_aten_randint_like_low_dtype(
    self: Tensor,
    low: Union[int, Tensor],
    high: Union[int, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.randint_like.low_dtype ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randint_like.out)
def conveter_aten_randint_like_out(
    self: Tensor,
    high: Union[int, Tensor],
    *,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint_like.out(Tensor self, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randint_like.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randint_like.low_dtype_out)
def conveter_aten_randint_like_low_dtype_out(
    self: Tensor,
    low: Union[int, Tensor],
    high: Union[int, Tensor],
    *,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint_like.low_dtype_out(Tensor self, SymInt low, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randint_like.low_dtype_out ge_converter is not implemented!")
