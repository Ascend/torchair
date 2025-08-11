from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.diag_embed.default)
def conveter_aten_diag_embed_default(
    self: Tensor,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.diag_embed.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.diag_embed.out)
def conveter_aten_diag_embed_out(
    self: Tensor,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::diag_embed.out(Tensor self, int offset=0, int dim1=-2, int dim2=-1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.diag_embed.out ge_converter is not implemented!")
