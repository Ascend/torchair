from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(3, 1), size=[3, 4]),
        Support(F16(3, 1, 2), size=[-1, 5, -1]),
        Support(F16(2, 3), size=[2, 2, 3]),
        Support(F16(3, 4, 2), size=[4, -1, -1, -1])
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.expand.default)
def conveter_aten_expand_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    *,
    implicit: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)"""
    if implicit:
        raise RuntimeError(
            "torch.ops.aten.expand.default ge_converter is not implemented when param implicit is True."
        )
    # performance optimization: if the input and output symbolic shape is equal, do not broadcast
    if hasattr(self, "_symsize") and meta_outputs is not None and hasattr(meta_outputs, "_symsize"):
        if str(self._symsize) == str(meta_outputs._symsize):
            return self
    return ge.BroadcastTo(self, size)

