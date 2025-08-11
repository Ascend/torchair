from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
     Support(F32(3, 2)),
     Support(U8(3, 2)),
     Support(U8(3, 2), 1),
     Support(U8(3, 2), 1, keepdim=True),
     Support(I8(3, 2)),
     Support(I8(3, 2), 1),
     Support(I8(3, 2), 1, keepdim=True),
     Support(F16(3, 2)),
     Support(F32(3, 2), 1),
     Support(F32(3, 2), 1, keepdim=True),
     Support(F16(3, 2), 1),
     Support(F16(3, 2), 1, keepdim=True),
     Support(F16(3, 2, 2), 2, keepdim=True),
])
@register_fx_node_ge_converter(torch.ops.aten.argmax.default)
def conveter_aten_argmax_default(
    self: Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    if dim is None:
        self_cp = ge.Reshape(self, dtype_promote((-1,), target_dtype=DataType.DT_INT64))
        return ge.ArgMaxV2(self_cp, 0)
    if keepdim:
        self_cp = dtype_promote(self, target_dtype=DataType.DT_FLOAT)
        index, _ = ge.ArgMaxWithValue(self_cp, dimension=dim, keep_dims=keepdim)
        index = dtype_promote(index, target_dtype=DataType.DT_INT64)
        return index
    dim = dtype_promote(dim, target_dtype=DataType.DT_INT64)
    return ge.ArgMaxV2(self, dim)


@register_fx_node_ge_converter(torch.ops.aten.argmax.out)
def conveter_aten_argmax_out(
    self: Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.argmax.out ge_converter is not supported!")
