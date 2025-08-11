from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.select.Dimname)
def conveter_aten_select_Dimname(
    self: Tensor, dim: str, index: int, meta_outputs: TensorSpec = None
):
    """NB: aten::select.Dimname(Tensor(a) self, str dim, int index) -> Tensor(a)"""
    raise RuntimeError(
        "torch.ops.aten.select.Dimname is redundant before pytorch 2.1.0,might be supported in future version.")


# TO DO: case2 will fail, fix view output case later
@declare_supported(
    [
        Support(F32(3, 4), dim=0, index=0),
        Support(F16(3, 4, 5), dim=1, index=2),
        Support(F16(3, 4, 5), dim=1, index=-1)
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.select.int)
def conveter_aten_select_int(
    self: Tensor, dim: int, index: Union[int, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)"""

    return ge.GatherV2(self, index, [dim], negative_index_support=True)