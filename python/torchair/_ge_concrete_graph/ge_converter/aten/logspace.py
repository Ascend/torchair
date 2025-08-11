from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.logspace.default)
def conveter_aten_logspace_default(
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    steps: int,
    base: float = 10.0,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::logspace(Scalar start, Scalar end, int steps, float base=10., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.logspace.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logspace.out)
def conveter_aten_logspace_out(
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    steps: int,
    base: float = 10.0,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::logspace.out(Scalar start, Scalar end, int steps, float base=10., *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logspace.out ge_converter is not implemented!")
