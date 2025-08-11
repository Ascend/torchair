from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 0, 5), 1, F32(2, 0, 5)),
    Support(I8(2, 0, 5), 1, I8(2, 0, 5))
])
@register_fx_node_ge_converter(torch.ops.aten.gather.default)
def conveter_aten_gather_default(
    self: Tensor,
    dim: int,
    index: Tensor,
    *,
    sparse_grad: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor"""
    index = dtype_promote(index, target_dtype=DataType.DT_INT64)
    return ge.GatherElements(self, index, dim=dim)


@register_fx_node_ge_converter(torch.ops.aten.gather.out)
def conveter_aten_gather_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    *,
    sparse_grad: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gather.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gather.dimname)
def conveter_aten_gather_dimname(
    self: Tensor,
    dim: str,
    index: Tensor,
    *,
    sparse_grad: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gather.dimname(Tensor self, str dim, Tensor index, *, bool sparse_grad=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.gather.dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gather.dimname_out)
def conveter_aten_gather_dimname_out(
    self: Tensor,
    dim: str,
    index: Tensor,
    *,
    sparse_grad: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gather.dimname_out(Tensor self, str dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gather.dimname_out ge_converter is not implemented!")
