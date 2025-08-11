from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(4, 2)),
        Support(I8(4, 2)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.t.default)
def conveter_aten_t_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::t(Tensor(a) self) -> Tensor(a)"""
    if self.rank < 2:
        return ge.Identity(self)
    elif self.rank == 2:
        perm = [1, 0]
        perm = dtype_promote(perm, target_dtype=DataType.DT_INT64)
        return ge.Transpose(self, perm)
    else:
        raise RuntimeError("torch.ops.aten.t.default unsupported case!")
