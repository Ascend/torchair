from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.index_put.default)
def conveter_aten_index_put_default(
    self: Tensor,
    indices: List[Optional[Tensor]],
    values: Tensor,
    accumulate: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.index_put.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_put.out)
def conveter_aten_index_put_out(
    self: Tensor,
    indices: List[Optional[Tensor]],
    values: Tensor,
    accumulate: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_put.out(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_put.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_put.hacked_twin)
def conveter_aten_index_put_hacked_twin(
    self: Tensor,
    indices: List[Tensor],
    values: Tensor,
    accumulate: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.index_put.hacked_twin ge_converter is not implemented!")
