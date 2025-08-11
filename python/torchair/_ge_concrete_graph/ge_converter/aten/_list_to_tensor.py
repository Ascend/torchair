from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._list_to_tensor.default)
def conveter_aten__list_to_tensor_default(self: List[int], meta_outputs: TensorSpec = None):
    """NB: aten::_list_to_tensor(int[] self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._list_to_tensor.default ge_converter is not implemented!")
