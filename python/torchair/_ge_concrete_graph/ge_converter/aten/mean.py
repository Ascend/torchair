from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 3, 2), dtype=torch.float16),
        Support(F32(2, 3, 2)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.mean.default)
def conveter_aten_mean_default(
    self: Tensor, *, dtype: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    if dtype is not None:
        self_cp = dtype_promote(self, target_dtype=dtype)
    else:
        self_cp = dtype_promote(self, target_dtype=meta_outputs.dtype)
    dims = [i for i in range(self.rank)]
    return ge.ReduceMean(self_cp, dims)


@declare_supported([
    Support(F32(12, 384, 32), [-1], True),
    Support(F32(12, 384, 32), [-1], False),
    Support(F32(12, 384, 32), [], True)
])
@register_fx_node_ge_converter(torch.ops.aten.mean.dim)
def conveter_aten_mean_dim(
    self: Tensor,
    dim: Optional[List[int]],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    if dim:
        dim_vec = dim
    else:
        dim_vec = [i for i in range(self.rank)]
    if dtype is not None:
        input_ge_dtype = torch_type_to_ge_type(dtype)
        return ge.ReduceMeanWithCast(self, dim_vec, keep_dims=keepdim, dtype=input_ge_dtype)
    return ge.ReduceMean(self, dim_vec, keep_dims=keepdim)


@register_fx_node_ge_converter(torch.ops.aten.mean.names_dim)
def conveter_aten_mean_names_dim(
    self: Tensor,
    dim: List[str],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::mean.names_dim(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    raise RuntimeError("torch.ops.aten.mean.names_dim is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.mean.names_out)
def conveter_aten_mean_names_out(
    self: Tensor,
    dim: List[str],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::mean.names_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.mean.names_out is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.mean.out)
def conveter_aten_mean_out(
    self: Tensor,
    dim: Optional[List[int]],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.mean.out is redundant before pytorch 2.1.0, "
                       "might be supported in future version.")
