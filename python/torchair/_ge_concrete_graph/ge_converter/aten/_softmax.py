from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(32, 100, 100), -1, False),
    Support(F32(32, 100, 100), 1, False),
])
@register_fx_node_ge_converter(torch.ops.aten._softmax.default)
def conveter_aten__softmax_default(
    self: Tensor, dim: int, half_to_float: bool, meta_outputs: TensorSpec = None
):
    """NB: aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""
    if half_to_float and self.dtype != DataType.DT_FLOAT16:
        raise RuntimeError(
            "torch.ops.aten._softmax.default: "
            "when half_to_tensor is True, input tensor must be half type.")
    return ge.SoftmaxV2(self, axes=[dim], half_to_float=half_to_float)


@register_fx_node_ge_converter(torch.ops.aten._softmax.out)
def conveter_aten__softmax_out(
    self: Tensor,
    dim: int,
    half_to_float: bool,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError(
        "torch.ops.aten._softmax.out is redundant before pytorch 2.1.0,"
        "it might be supported in future version.")


try:
    op = torch.ops.aten._safe_softmax.default
except (ImportError, AttributeError):
    op = None
if op is not None:
    @declare_supported([
        Support(F32(32, 100, 100), -1, torch.float16),
        Support(F32(32, 100, 100), 1, torch.bfloat16),
        Support(F32(32, 100, 100), 1, torch.float),
        Support(F32(32, 100, 100), 1, torch.float64),
    ])
    @register_fx_node_ge_converter(torch.ops.aten._safe_softmax.default)
    def conveter_aten__safe_softmax_default(
        self: Tensor,
        dim: int, 
        dtype: Optional[int] = None,
        meta_outputs: TensorSpec = None
    ):
        """NB: aten::_safe_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"""
        target_dtype = dtype if dtype else meta_outputs.dtype
        self = dtype_promote(self, target_dtype=target_dtype)
        return ge.SoftmaxV2(self, axes=[dim], half_to_float=False)