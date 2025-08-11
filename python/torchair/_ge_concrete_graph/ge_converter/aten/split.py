from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(7, 2), 2),
    Support(F16(8, 2), 2),
    Support(F16(7, 2), 4)
])
@register_fx_node_ge_converter(torch.ops.aten.split.Tensor)
def conveter_aten_split_Tensor(
    self: Tensor, split_size: Union[int, Tensor], dim: int = 0, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]"""
    if dim > 2147483647:
        dim = dtype_promote(dim, target_dtype=DataType.DT_INT64)
    else:
        dim = dtype_promote(dim, target_dtype=DataType.DT_INT32)
    split_sizes = [split_size for _ in range(len(meta_outputs))]
    split_sizes[-1] = -1
    split_sizes = ge.Pack(split_sizes, N=len(meta_outputs))
    split_sizes = dtype_promote(split_sizes, target_dtype=DataType.DT_INT64)
    return ge.SplitV(self, split_sizes, dim, num_split=len(meta_outputs))


@register_fx_node_ge_converter(torch.ops.aten.split.sizes)
def conveter_aten_split_sizes(
    self: Tensor,
    split_size: Union[List[int], Tensor],
    dim: int = 0,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::split.sizes(Tensor(a -> *) self, SymInt[] split_size, int dim=0) -> Tensor(a)[]"""
    raise NotImplementedError("torch.ops.aten.split.sizes ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.split.str)
def conveter_aten_split_str(
    self: str, separator: Optional[str] = None, max: int = -1, meta_outputs: TensorSpec = None
):
    """NB: aten::split.str(str self, str? separator=None, int max=-1) -> str[]"""
    raise NotImplementedError("torch.ops.aten.split.str ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.split.default)
def conveter_aten_split_default(
    self: Tensor, split_sizes: List[int], dim: int = 0, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::split(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> Tensor(a)[]"""
    raise NotImplementedError("torch.ops.aten.split.default ge_converter is not implemented!")
