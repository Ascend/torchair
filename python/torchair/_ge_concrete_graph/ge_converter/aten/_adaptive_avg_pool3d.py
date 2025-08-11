from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._adaptive_avg_pool3d.default)
def conveter_aten__adaptive_avg_pool3d_default(
    self: Tensor, output_size: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::_adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._adaptive_avg_pool3d.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._adaptive_avg_pool3d.out)
def conveter_aten__adaptive_avg_pool3d_out(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_adaptive_avg_pool3d.out(Tensor self, SymInt[3] output_size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._adaptive_avg_pool3d.out ge_converter is not implemented!")
