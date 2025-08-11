from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._assert_async.default)
def conveter_aten__assert_async_default(self: Tensor):
    """NB: aten::_assert_async(Tensor self) -> ()"""
    raise NotImplementedError("torch.ops.aten._assert_async.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._assert_async.msg)
def conveter_aten__assert_async_msg(
    self: Tensor, assert_msg: str):
    """NB: aten::_assert_async.msg(Tensor self, str assert_msg) -> ()"""
    raise NotImplementedError("torch.ops.aten._assert_async.msg ge_converter is not implemented!")
