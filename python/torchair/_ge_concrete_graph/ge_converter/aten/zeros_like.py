from torchair._ge_concrete_graph.ge_converter.converter_utils import *

    
@declare_supported(
    [
        Support(F32(8, 8), dtype=torch.int32),
        Support(F32(8, 8)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.zeros_like.default)
def conveter_aten_zeros_like_default(
    self: Tensor,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    if dtype is not None:
        self = dtype_promote(self, target_dtype=dtype)
    return ge.ZerosLike(self)


@register_fx_node_ge_converter(torch.ops.aten.zeros_like.out)
def conveter_aten_zeros_like_out(
    self: Tensor,
    *,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::zeros_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.zeros_like.out ge_converter is not supported!")
