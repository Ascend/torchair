import operator
from functools import wraps, reduce, lru_cache
from typing import Callable, Optional
import torch
from torch import Tensor
from torch._ops import OpOverload, OpOverloadPacket
from torch._subclasses import fake_tensor as _subclasses_fake_tensor
from torch._C import DispatchKey
from torch._prims_common.wrappers import out_wrapper
from torch._decomp import get_decompositions, decomposition_table

aten = torch.ops.aten


def run_once(f):
    """Runs a function (successfully) only once.
    The running can be reset by setting the `has_run` attribute to False
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            result = f(*args, **kwargs)
            wrapper.has_run = True
            return result
        return None
    wrapper.has_run = False
    return wrapper


npu_meta_table = {}


def _add_op_to_meta_table(op, fn):
    overloads = []
    if isinstance(op, OpOverload):
        overloads.append(op)
    else:
        assert isinstance(op, OpOverloadPacket)
        for ol in op.overloads():
            overloads.append(getattr(op, ol))

    for op_overload in overloads:
        if op_overload in npu_meta_table:
            raise RuntimeError(f"duplicate registrations for npu_meta_table {op_overload}")
        npu_meta_table[op_overload] = fn


def register_meta_npu(op):
    def meta_decorator(fn: Callable):
        _add_op_to_meta_table(op, fn)
        return fn

    return meta_decorator


@register_meta_npu(aten.native_dropout)
def meta_native_dropout(tensor_input: Tensor, p: float, train: Optional[bool]):
    if train and p != 0:
        sizes_1 = tensor_input.shape
        numel = reduce(operator.mul, sizes_1)
        numel = (numel + 128 - 1) // 128 * 128
        numel = numel // 8
        return (torch.empty_like(tensor_input), torch.empty(numel, dtype=torch.uint8, device=tensor_input.device))
    else:
        return (tensor_input, torch.ones_like(tensor_input, dtype=torch.bool))


@register_meta_npu(aten.native_dropout_backward)
def meta_native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: float):
    return torch.empty_like(grad_output)


def patch_torch_decomp_decompositions():
    '''
    Because source torch_decomp_decompositions only enable the decompositions in
    torch/_decomp/decompositions.py. Patch it to make decompositions in this file work.
    '''
    src_func = _subclasses_fake_tensor.torch_decomp_decompositions
    @lru_cache(None)
    def torch_decomp_decompositions_new(func):
        if func in npu_meta_table.keys():
            return True
        return src_func(func)
    _subclasses_fake_tensor.torch_decomp_decompositions = torch_decomp_decompositions_new


def npu_patch_meta():
    '''
    Torch official register decompostions and meta func for some aten ops,
    which will raise conflict when npu outputs' dtype and shape are different
    from native impl. Delete decompositions and meta func of these ops and add
    npu decompositions and meta func.
    '''
    for op_overload, fn in npu_meta_table.items():
        assert isinstance(op_overload, OpOverload)
        decomposition_table[op_overload] = fn
        op_overload.py_kernels.pop(DispatchKey.Meta, None)
        op_overload.py_impl(DispatchKey.Meta)(fn)

    patch_torch_decomp_decompositions()



def disable_implicit_decomposition():
    '''
    Since torch official will implicitly decompose some aten ops,
    disable some ops here to avoid poor performance after decompose.
    '''
    disable_aten_ops = [
        'aten.upsample_nearest1d.vec', 'aten.upsample_nearest1d.default',
        'aten.upsample_nearest2d.vec', 'aten.upsample_nearest2d.default',
        'aten.upsample_nearest3d.vec', 'aten.upsample_nearest3d.default',
    ]

    for op_override in decomposition_table.keys():
        if str(op_override) in disable_aten_ops:
            if DispatchKey.Autograd in op_override.py_kernels:
                op_override.py_kernels.pop(DispatchKey.Autograd)
            if DispatchKey.CompositeImplicitAutograd in op_override.py_kernels:
                op_override.py_kernels.pop(DispatchKey.CompositeImplicitAutograd)


def register_matmul_backward_decomp():
    '''
    Torch_npu currently dispatch linear to matmul and matmul_backward
    instead of mm and mm_backwrd. This will lead to some bug in npu_backend
    and the decomposition can fix it.
    '''
    @aten.matmul_backward.default.py_impl(DispatchKey.CompositeImplicitAutograd)
    @out_wrapper("grad_self", "grad_other")
    def matmul_backward_decomposition(grad, self, other, mask):
        dim_self = self.dim()
        dim_other = other.dim()

        size_grad = grad.size()
        size_self = self.size()
        size_other = other.size()
        grad_self = None
        grad_other = None
        if dim_self == 1 and dim_other == 1:
            grad_self = other.mul(grad) if mask[0] else grad_self
            grad_other = self.mul(grad) if mask[1] else grad_other
        elif dim_self == 2 and dim_other == 1:
            grad_self = grad.unsqueeze(1).mm(other.unsqueeze(0)) if mask[0] else grad_self
            grad_other = self.transpose(-1, -2).mm(grad.unsqueeze(1)).squeeze_(1) if mask[1] else grad_other
        elif dim_self == 1 and dim_other == 2:
            grad_self = grad.unsqueeze(0).mm(other.transpose(-1, -2)).squeeze_(0) if mask[0] else grad_self
            grad_other = self.unsqueeze(1).mm(grad.unsqueeze(0)) if mask[1] else grad_other
        elif dim_self >= 3 and (dim_other == 1 or dim_other == 2):
            # create a 2D-matrix from grad
            view_size = 1 if dim_other == 1 else size_grad[-1]
            unfolded_grad = (grad.unsqueeze(-1) if dim_other == 1 else grad).contiguous().view(-1, view_size)
            if mask[0]:
                unfolded_other = other.unsqueeze(0) if dim_other == 1 else other.transpose(-1, -2)
                grad_self = unfolded_grad.mm(unfolded_other).view(size_self)

            if mask[1]:
                # create a 2D-matrix from self
                unfolded_self = self.contiguous().view(-1, size_self[-1])
                grad_other = unfolded_self.transpose(-1, -2).mm(unfolded_grad).view(size_other)
        elif (dim_self == 1 or dim_self == 2) and dim_other >= 3:
            # create a 2D-matrix from grad
            view_size = 1 if dim_self == 1 else size_grad[size_grad.size() - 2]
            unfolded_grad_t = grad.view(-1, view_size) if dim_self == 1 else \
                                                            grad.transpose(-1, -2).contiguous().view(-1, view_size)
            if mask[0]:
                # create a 2D-matrix from other
                unfolded_other_t = \
                    other.transpose(-1, -2).contiguous().view(-1, size_other[-2]).transpose(-1, -2)
                grad_self = unfolded_other_t.mm(unfolded_grad_t).transpose(-1, -2).view(size_self)

            if mask[1]:
                size_other_t = size_other[:-2]
                size_other_t.extend([size_other[dim_other - 1], size_other[dim_other - 2]])
                unfolded_self = self.unsqueeze(0) if dim_self == 1 else self
                grad_other = unfolded_grad_t.mm(unfolded_self).view(size_other_t).transpose(-1, -2)
        else:
            grad_self = torch.matmul(grad, other.transpose(-1, -2)) if mask[0] else grad_self
            grad_other = torch.matmul(self.transpose(-1, -2), grad) if mask[1] else grad_other

        return grad_self, grad_other


@run_once
def process_npu_decomposition():
    disable_implicit_decomposition()
    register_matmul_backward_decomp()
    npu_patch_meta()