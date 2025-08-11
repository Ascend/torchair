from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.index_fill_.int_Tensor)
def conveter_aten_index_fill__int_Tensor(
    self: Tensor, dim: int, index: Tensor, value: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_fill_.int_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill_.int_Scalar)
def conveter_aten_index_fill__int_Scalar(
    self: Tensor,
    dim: int,
    index: Tensor,
    value: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_fill_.int_Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill_.Dimname_Scalar)
def conveter_aten_index_fill__Dimname_Scalar(
    self: Tensor,
    dim: str,
    index: Tensor,
    value: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_fill_.Dimname_Scalar(Tensor(a!) self, str dim, Tensor index, Scalar value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_fill_.Dimname_Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill_.Dimname_Tensor)
def conveter_aten_index_fill__Dimname_Tensor(
    self: Tensor, dim: str, index: Tensor, value: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_fill_.Dimname_Tensor(Tensor(a!) self, str dim, Tensor index, Tensor value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_fill_.Dimname_Tensor ge_converter is not implemented!")
