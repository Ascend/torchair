from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.topk.default)
def conveter_aten_topk_default(
    self: Tensor,
    k: Union[int, Tensor],
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)"""
    value, indices = ge.TopKV2(self, k, sorted=sorted, dim=dim, largest=largest)
    indices = dtype_promote(indices, target_dtype=meta_outputs[1].dtype)
    return value, indices


@register_fx_node_ge_converter(torch.ops.aten.topk.values)
def conveter_aten_topk_values(
    self: Tensor,
    k: Union[int, Tensor],
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    values: Tensor = None,
    indices: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::topk.values(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)"""
    raise NotImplementedError("torch.ops.aten.topk.values ge_converter is not implemented!")
