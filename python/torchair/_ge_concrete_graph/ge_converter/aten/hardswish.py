from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2)),
    Support(F16(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.hardswish.default)
def conveter_aten_hardswish_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::hardswish(Tensor self) -> Tensor"""
    return ge.HardSwish(self)


@register_fx_node_ge_converter(torch.ops.aten.hardswish.out)
def conveter_aten_hardswish_out(
        self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::hardswish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError(
        "torch.ops.aten.hardswish.out is redundant before pytorch 2.1.0,might be supported in future version.")
