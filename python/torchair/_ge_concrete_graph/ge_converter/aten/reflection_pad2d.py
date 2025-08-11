from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(1, 32, 64, 64), [1, 1, 1, 1]),
        Support(F32(1, 3, 32, 32), [3, 3, 3, 3]),
        Support(F32(1, 3, 3), [1, 1, 1, 1]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.reflection_pad2d.default)
def conveter_aten_reflection_pad2d_default(
    self: Tensor, padding: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor"""
    if isinstance(padding, Tensor):
        raise NotImplementedError("When padding is Tensor, torch.ops.aten.reflection_pad2d.default ge_converter is not implemented!")
    if len(padding) != 4:
        raise AssertionError("padding size is expected to be 4")
    self_cp = ge.Unsqueeze(self, axes=[0]) if self.rank == 3 else self
    pads = [0] * (2 * 4) if self.rank == 3 else [0] * (2 * self.rank)
    if len(padding) <= len(pads):
        pads[0: len(padding)] = padding
    else:
        pads = padding[0: len(pads)]
    paddings = []
    pads_len = len(pads)
    while pads_len > 1:
        paddings.append(pads[pads_len - 2])
        paddings.append(pads[pads_len - 1])
        pads_len -= 2
    self_rank = 4 if self.rank == 3 else self.rank
    paddings = ge.Reshape(paddings, [self_rank, 2])
    result = ge.MirrorPad(self_cp, paddings, mode="REFLECT")
    if self.rank == 3:
        result = ge.Squeeze(result, axis=[0])
    return result


@register_fx_node_ge_converter(torch.ops.aten.reflection_pad2d.out)
def conveter_aten_reflection_pad2d_out(
    self: Tensor,
    padding: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::reflection_pad2d.out(Tensor self, SymInt[4] padding, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.reflection_pad2d.out ge_converter is not implemented!")
