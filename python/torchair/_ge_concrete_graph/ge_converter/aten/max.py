from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.max.other)
def conveter_aten_max_other(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::max.other(Tensor self, Tensor other) -> Tensor"""
    raise RuntimeError("torch.ops.aten.max.other ge_converter is not supported!")


@declare_supported([
    Support(I32(2, 2)),
    Support(F32(2, 2)),
    Support(F16(3, 4)),
    Support(U8(3, 4))
])
@register_fx_node_ge_converter(torch.ops.aten.max.default)
def conveter_aten_max_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::max(Tensor self) -> Tensor"""
    dim = list(range(self.rank))
    dim = dtype_promote(dim, target_dtype=DataType.DT_INT64)
    self = dtype_promote(self, target_dtype=meta_outputs.dtype)
    return ge.ReduceMax(self, dim)


@declare_supported([
    Support(F32(4, 4), 1, False),
    Support(F32(4, 4), 0, True),
])
@register_fx_node_ge_converter(torch.ops.aten.max.dim)
def conveter_aten_max_dim(
    self: Tensor, dim: int, keepdim: bool = False, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"""
    index, output = ge.ArgMaxWithValue(self, dimension=dim, keep_dims=keepdim)
    index = dtype_promote(index, target_dtype=meta_outputs[1].dtype)
    return output, index


@register_fx_node_ge_converter(torch.ops.aten.max.dim_max)
def conveter_aten_max_dim_max(
    self: Tensor,
    dim: int,
    keepdim: bool = False,
    *,
    max: Tensor = None,
    max_values: Tensor = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)"""
    raise RuntimeError("torch.ops.aten.max.dim_max ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.max.names_dim)
def conveter_aten_max_names_dim(
    self: Tensor, dim: str, keepdim: bool = False, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::max.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)"""
    raise RuntimeError("torch.ops.aten.max.names_dim ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.max.names_dim_max)
def conveter_aten_max_names_dim_max(
    self: Tensor,
    dim: str,
    keepdim: bool = False,
    *,
    max: Tensor = None,
    max_values: Tensor = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::max.names_dim_max(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)"""
    raise RuntimeError("torch.ops.aten.max.names_dim_max ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.max.unary_out)
def conveter_aten_max_unary_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::max.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.max.unary_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.max.out)
def conveter_aten_max_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.max.out ge_converter is not supported!")
