from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support([F32(2, 2), F32(2, 1)], dim=1),
        Support([F32(2, 2), F32(1, 2)], dim=0),
        Support([F32(2, 2), F16(1, 2)], dim=0),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.cat.default)
def conveter_aten_cat_default(
    tensors: List[Tensor], dim: int = 0, meta_outputs: TensorSpec = None
):
    """NB: aten::cat(Tensor[] tensors, int dim=0) -> Tensor"""
    tensors = [dtype_promote(arg, target_dtype=meta_outputs.dtype) for arg in tensors]
    return ge.ConcatV2(tensors, concat_dim=dim, N=len(tensors))


@register_fx_node_ge_converter(torch.ops.aten.cat.names)
def conveter_aten_cat_names(tensors: List[Tensor], dim: str, meta_outputs: TensorSpec = None):
    """NB: aten::cat.names(Tensor[] tensors, str dim) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.cat.names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cat.names_out)
def conveter_aten_cat_names_out(
    tensors: List[Tensor], dim: str, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cat.names_out(Tensor[] tensors, str dim, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cat.names_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cat.out)
def conveter_aten_cat_out(
    tensors: List[Tensor], dim: int = 0, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cat.out ge_converter is not implemented!")
