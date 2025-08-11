from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.index.Tensor)
def conveter_aten_index_Tensor(
    self: Tensor, indices: List[Optional[Tensor]], meta_outputs: TensorSpec = None
):
    """NB: aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor"""
    mask = [1 if indice else 0 for indice in indices]
    indices = [dtype_promote(i, target_dtype=DataType.DT_INT64) for i in indices if i]
    return ge.IndexByTensor(self, indices, indices_mask=mask)


@register_fx_node_ge_converter(torch.ops.aten.index.Tensor_out)
def conveter_aten_index_Tensor_out(
    self: Tensor,
    indices: List[Optional[Tensor]],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index.Tensor_out(Tensor self, Tensor?[] indices, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.index.Tensor_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.index.Tensor_hacked_twin)
def conveter_aten_index_Tensor_hacked_twin(
    self: Tensor, indices: List[Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> Tensor"""
    raise RuntimeError("torch.ops.aten.index.Tensor_hacked_twin ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.index.str)
def conveter_aten_index_str(
    self: str, substr: str, start: int = 0, end: int = -1, meta_outputs: TensorSpec = None
):
    """NB: aten::index.str(str self, str substr, int start=0, int end=-1) -> int"""
    raise RuntimeError("torch.ops.aten.index.str ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.index.list_int)
def conveter_aten_index_list_int(self: List[int], el: int, meta_outputs: TensorSpec = None):
    """NB: aten::index.list_int(int[] self, int el) -> int"""
    raise RuntimeError("torch.ops.aten.index.list_int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.index.list_float)
def conveter_aten_index_list_float(
    self: List[float], el: float, meta_outputs: TensorSpec = None
):
    """NB: aten::index.list_float(float[] self, float el) -> int"""
    raise RuntimeError("torch.ops.aten.index.list_float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.index.list_bool)
def conveter_aten_index_list_bool(self: List[bool], el: bool, meta_outputs: TensorSpec = None):
    """NB: aten::index.list_bool(bool[] self, bool el) -> int"""
    raise RuntimeError("torch.ops.aten.index.list_bool ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.index.list_Tensor)
def conveter_aten_index_list_Tensor(
    self: List[Tensor], el: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index.list_Tensor(Tensor[] self, Tensor el) -> int"""
    raise RuntimeError("torch.ops.aten.index.list_Tensor ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.index.list_str)
def conveter_aten_index_list_str(self: List[str], el: str, meta_outputs: TensorSpec = None):
    """NB: aten::index.list_str(str[] self, str el) -> int"""
    raise RuntimeError("torch.ops.aten.index.list_str ge_converter is not supported!")
