from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.empty_like.default)
def conveter_aten_empty_like_default(
    self: Tensor,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    
    if dtype is None:
        dtype = self.dtype

    if layout is not None and (layout != torch.strided):
        raise RuntimeError("torch.ops.aten.empty_like.default is only supported on dense tensor now.")
    
    if memory_format is not None and memory_format != torch.contiguous_format \
            and memory_format != torch.preserve_format:
        raise RuntimeError("torch.ops.aten.empty_like.default is only supported "
                "contiguous_format and preserve_format now.")
    # There is a bug with the op Empty when dynamic=True and dtype=int8.
    # So replace Empty with Fill.
    return ge.Fill(ge.Shape(self), ge.Cast(0., dst_type=meta_outputs.dtype))


@register_fx_node_ge_converter(torch.ops.aten.empty_like.out)
def conveter_aten_empty_like_out(
    self: Tensor,
    *,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::empty_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.empty_like.out ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")
