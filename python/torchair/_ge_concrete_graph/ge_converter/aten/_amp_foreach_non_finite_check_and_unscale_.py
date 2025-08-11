from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._amp_foreach_non_finite_check_and_unscale_.default)
def conveter_aten__amp_foreach_non_finite_check_and_unscale__default(
    self: List[Tensor], found_inf: Tensor, inv_scale: Tensor):
    """NB: aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self, Tensor(b!) found_inf, Tensor inv_scale) -> ()"""
    raise NotImplementedError(
        "torch.ops.aten._amp_foreach_non_finite_check_and_unscale_.default ge_converter is not implemented!"
    )
