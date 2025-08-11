from torchair._ge_concrete_graph.ge_converter.converter_utils import *
try:
    from torch import sym_sqrt
except ImportError:
    from torch.fx.experimental.symbolic_shapes import sym_sqrt


@register_fx_node_ge_converter(sym_sqrt)
def conveter_sym_sqrt(
        self: Union[Number, Tensor],
        meta_outputs: TensorSpec = None
):
    if not isinstance(self, Tensor):
        return sqrt(self)
    return ge.Sqrt(self)