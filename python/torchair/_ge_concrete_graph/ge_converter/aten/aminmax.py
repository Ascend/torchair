from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.aminmax.default)
def conveter_aten_aminmax_default(
    self: Tensor,
    *,
    dim: Optional[int] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)"""
    raise NotImplementedError("torch.ops.aten.aminmax.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.aminmax.out)
def conveter_aten_aminmax_out(
    self: Tensor,
    *,
    dim: Optional[int] = None,
    keepdim: bool = False,
    min: Tensor = None,
    max: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)"""
    raise NotImplementedError("torch.ops.aten.aminmax.out ge_converter is not implemented!")
