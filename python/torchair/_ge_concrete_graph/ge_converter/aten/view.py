from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 2), [4]),
        Support(F16(16), [2, 8]),
        Support(U8(16), [2, 8])
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.view.default)
def conveter_aten_view_default(
    self: Tensor, size: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)"""
    # Use dtype_promote in specific case to reduce the number of Cast operators
    if isinstance(size, list):
        size = dtype_promote(size, target_dtype=DataType.DT_INT64)
    elif size.dtype != DataType.DT_INT64 and size.dtype != DataType.DT_INT32:
        size = dtype_promote(size, target_dtype=DataType.DT_INT64)
    return ge.Reshape(self, size)


@register_fx_node_ge_converter(torch.ops.aten.view.dtype)
def conveter_aten_view_dtype(self: Tensor, dtype: int, meta_outputs: TensorSpec = None):
    """NB: aten::view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.view.dtype ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.reshape.default)
def conveter_aten_reshape_default(
    self: Tensor, shape: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)"""
    # Use dtype_promote in specific case to reduce the number of Cast operators
    if isinstance(shape, list):
        shape = dtype_promote(shape, target_dtype=DataType.DT_INT64)
    elif shape.dtype != DataType.DT_INT64 and shape.dtype != DataType.DT_INT32:
        shape = dtype_promote(shape, target_dtype=DataType.DT_INT64)
    return ge.Reshape(self, shape)
