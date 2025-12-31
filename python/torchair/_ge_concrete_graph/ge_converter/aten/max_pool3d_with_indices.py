from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.max_pool3d_with_indices.default)
def conveter_aten_max_pool3d_with_indices_default(
    self: Tensor,
    kernel_size: List[int],
    stride: List[int] = [],
    padding: List[int] = [0, 0, 0],
    dilation: List[int] = [1, 1, 1],
    ceil_mode: bool = False,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)"""
    """ This converter is a stopgap measure designed to avoid a series issues caused by the imcompatibility between the CANN IR 'MaxPoolWithArgmaxV1' and 
        the aten IR 'max_pool3d_with_indices_backward'. Therefore, no testcast will be set and cannot be set. """
    assert_args_checkout(len(kernel_size) == 1 or len(kernel_size) == 3,
                         "torch.ops.aten.max_pool3d_with_indices.default: kernel_size must either be a single int, "
                         "or a tuple of three ints")
    assert_args_checkout(len(stride) == 0 or len(stride) == 1 or len(stride) == 3,
                         "torch.ops.aten.max_pool3d_with_indices.default: stride must either be omitted, a single "
                         "int, or a tuple of three ints")
    assert_args_checkout(len(padding) == 1 or len(padding) == 3,
                         "torch.ops.aten.max_pool3d_with_indices.default: padding must either be a single int, "
                         "or a tuple of three ints")
    assert_args_checkout(len(dilation) == 1 or len(dilation) == 3,
                         "torch.ops.aten.max_pool3d_with_indices.default: dilation must either be a single int, "
                         "or a tuple of three ints")
    k_d = kernel_size[0]
    k_h = k_d if len(kernel_size) == 1 else kernel_size[1]
    k_w = k_d if len(kernel_size) == 1 else kernel_size[2]
    kernel_sizes = [k_d, k_h, k_w]
    s_d = k_d if len(stride) == 0 else stride[0]
    s_h = k_h if len(stride) == 0 else s_d if len(stride) == 1 else stride[1]
    s_w = k_w if len(stride) == 0 else s_d if len(stride) == 1 else stride[2]
    strides = [s_d, s_h, s_w]
    pad_d = padding[0]
    pad_h = pad_d if len(padding) == 1 else padding[1]
    pad_w = pad_d if len(padding) == 1 else padding[2]
    paddings = [pad_d, pad_h, pad_w]
    dil_d = dilation[0]
    dil_h = dil_d if len(dilation) == 1 else dilation[1]
    dil_w = dil_d if len(dilation) == 1 else dilation[2]
    dilations = [dil_d, dil_h, dil_w]
    input_dim = self.rank
    if input_dim == 4:
        self = ge.Unsqueeze(self, axes=[0])
    output, argmax = ge.MaxPool3DWithArgmaxV2(self, ksize=kernel_sizes, strides=strides, pads=paddings, \
                                              dilation=dilations, ceil_mode=ceil_mode)
    specific_op_input_layout(output, indices=0, layout="NCDHW")
    specific_op_output_layout(output, indices=[0, 1], layout="NCDHW")
    argmax = dtype_promote(argmax, target_dtype=torch.int64)
    if input_dim == 4:
        output = ge.Squeeze(output, axis=[0])
        argmax = ge.Squeeze(argmax, axis=[0])
    return output, argmax


@register_fx_node_ge_converter(torch.ops.aten.max_pool3d_with_indices.out)
def conveter_aten_max_pool3d_with_indices_out(
    self: Tensor,
    kernel_size: List[int],
    stride: List[int] = [],
    padding: List[int] = [0, 0, 0],
    dilation: List[int] = [1, 1, 1],
    ceil_mode: bool = False,
    *,
    out: Tensor = None,
    indices: Tensor = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::max_pool3d_with_indices.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))"""
    raise NotImplementedError("torch.ops.aten.max_pool3d_with_indices.out ge_converter is not implemented!")
