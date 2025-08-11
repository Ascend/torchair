from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.diagonal.default)
def conveter_aten_diagonal_default(
    self: Tensor,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.diagonal.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.diagonal.Dimname)
def conveter_aten_diagonal_Dimname(
    self: Tensor,
    *,
    outdim: str,
    dim1: str,
    dim2: str,
    offset: int = 0,
    meta_outputs: TensorSpec = None
):
    """NB: aten::diagonal.Dimname(Tensor(a) self, *, str outdim, str dim1, str dim2, int offset=0) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.diagonal.Dimname ge_converter is not implemented!")
