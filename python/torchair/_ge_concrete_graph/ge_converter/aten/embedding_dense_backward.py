from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.embedding_dense_backward.default)
def conveter_aten_embedding_dense_backward_default(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: Union[int, Tensor],
    padding_idx: Union[int, Tensor],
    scale_grad_by_freq: bool,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor"""
    if isinstance(num_weights, Tensor) or isinstance(padding_idx, Tensor):
        raise RuntimeError("ge.EmbeddingDenseGrad is not support when num_weights or padding_idx is Tensor")
    if scale_grad_by_freq:
        indices = ge.Cast(indices, dst_type=DataType.DT_INT32)
    return ge.EmbeddingDenseGrad(grad_output, indices,
                                    num_weights=num_weights,
                                    padding_idx=padding_idx,
                                    scale_grad_by_freq=scale_grad_by_freq)


@register_fx_node_ge_converter(torch.ops.aten.embedding_dense_backward.out)
def conveter_aten_embedding_dense_backward_out(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: Union[int, Tensor],
    padding_idx: Union[int, Tensor],
    scale_grad_by_freq: bool,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::embedding_dense_backward.out(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.embedding_dense_backward.out ge_converter is not implemented!")
