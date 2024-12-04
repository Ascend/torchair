import torch
from torch._C import DispatchKey
from torch._decomp import decomposition_table
from torch._decomp.decompositions import upsample_compute_output_size, get_scale_value


def new_upsample_nearest2d_vec(input_tensor, output_size, scale_factors):
    osize = upsample_compute_output_size(
        input_tensor.size(), output_size, scale_factors)
    scale_h = get_scale_value(scale_factors, 0)
    scale_w = get_scale_value(scale_factors, 1)

    return torch.ops.aten.upsample_nearest2d(input_tensor, osize, scale_h, scale_w)


'''
Since torch official will implicitly decompose some aten ops,
replace and disable some ops here to avoid poor performance after decompose.
'''
replace_aten_ops = {
    'aten.upsample_nearest2d.vec': new_upsample_nearest2d_vec,
}

disable_aten_ops = [
    'aten.upsample_nearest2d.default',
]


def adjust_implicit_decomposition():
    for op_override in decomposition_table.keys():
        if str(op_override) in disable_aten_ops:
            if DispatchKey.Autograd in op_override.py_kernels:
                op_override.py_kernels.pop(DispatchKey.Autograd)
            if DispatchKey.CompositeImplicitAutograd in op_override.py_kernels:
                op_override.py_kernels.pop(
                    DispatchKey.CompositeImplicitAutograd)
        if str(op_override) in replace_aten_ops:
            if DispatchKey.Autograd in op_override.py_kernels:
                op_override.py_kernels[DispatchKey.Autograd] = replace_aten_ops[str(
                    op_override)]
            if DispatchKey.CompositeImplicitAutograd in op_override.py_kernels:
                op_override.py_kernels[DispatchKey.CompositeImplicitAutograd] = replace_aten_ops[str(
                    op_override)]
