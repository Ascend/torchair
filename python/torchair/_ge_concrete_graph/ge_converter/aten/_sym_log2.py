from torchair._ge_concrete_graph.ge_converter.converter_utils import *


try:
    op = torch._sym_log2
except (ImportError, AttributeError):
    op = None
if op is not None:
    @register_fx_node_ge_converter(torch._sym_log2)
    def conveter_aten_sym_log2(self: Tensor, meta_outputs: TensorSpec = None):
        """NB: aten::_sym_log2(Tensor self) -> Tensor"""
        return ge.Log(self, base=2)