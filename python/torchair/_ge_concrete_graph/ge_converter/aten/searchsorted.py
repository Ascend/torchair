from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(I64(4), I64(3)),
    Support(I64(4), I64(3), right=True),
    Support(I64(4), I64(3), out_int32=True, right=True),
    Support(I64(4), I64(3), out_int32=True, side="left"),
    Support(I64(5), I64(3), sorter=I64(5)),
    Support(I64(2, 4), I64(2, 3)),
])
@register_fx_node_ge_converter(torch.ops.aten.searchsorted.Tensor)
def conveter_aten_searchsorted_Tensor(
    sorted_sequence: Tensor,
    self: Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: Optional[str] = None,
    sorter: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor"""
    dst_dtype = DataType.DT_INT32 if out_int32 else DataType.DT_INT64

    if side is not None:
        if side not in ("left", "right"):
            raise ValueError(f"torch.ops.aten.searchsorted.Tensor side must be 'left' or 'right', got {side!r}")
        right = (side == "right")

    return ge.SearchSorted(sorted_sequence, self, sorter=sorter, dtype=dst_dtype, right=right)


@register_fx_node_ge_converter(torch.ops.aten.searchsorted.Tensor_out)
def conveter_aten_searchsorted_Tensor_out(
    sorted_sequence: Tensor,
    self: Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: Optional[str] = None,
    sorter: Optional[Tensor] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.searchsorted.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.searchsorted.Scalar)
def conveter_aten_searchsorted_Scalar(
    sorted_sequence: Tensor,
    self: Union[Number, Tensor],
    *,
    out_int32: bool = False,
    right: bool = False,
    side: Optional[str] = None,
    sorter: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.searchsorted.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.searchsorted.Scalar_out)
def conveter_aten_searchsorted_Scalar_out(
    sorted_sequence: Tensor,
    self: Union[Number, Tensor],
    *,
    out_int32: bool = False,
    right: bool = False,
    side: Optional[str] = None,
    sorter: Optional[Tensor] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::searchsorted.Scalar_out(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.searchsorted.Scalar_out ge_converter is not implemented!")
