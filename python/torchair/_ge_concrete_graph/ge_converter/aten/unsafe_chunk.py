from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.unsafe_chunk.default)
def conveter_aten_unsafe_chunk_default(
    self: Tensor, chunks: int, dim: int = 0, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.unsafe_chunk.default ge_converter is not implemented!")
