import os
import sys
import operator
from functools import wraps, reduce, lru_cache
from typing import Callable, Optional, Tuple
import torch
from torch import Tensor
from torch._ops import OpOverload, OpOverloadPacket
from torch._subclasses import fake_tensor as _subclasses_fake_tensor
from torch._C import DispatchKey
from torch._refs import div as refs_div, _broadcast_shapes
from torch._prims_common import corresponding_real_dtype, corresponding_complex_dtype
from torch._prims_common.wrappers import out_wrapper
from torch._decomp import decomposition_table, decompositions_for_rng, get_decompositions
from torch._dynamo.symbolic_convert import break_graph_if_unsupported, InstructionTranslatorBase, stack_op
from torch._dynamo.exc import Unsupported
from torch._dynamo.variables.lists import TupleVariable
from torch._dynamo.variables.nn_module import NNModuleVariable
from .adjust_implicit_decomposition import adjust_implicit_decomposition
from .adjust_traceable_collective_remaps import adjust_traceable_collective_remaps


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
break_fn_table = {}
break_mapping_table = {}
avoid_make_fallback_table = []


def _add_op_to_meta_table(op, fn, avoid_fallback_flag=False):
    overloads = []
    if isinstance(op, OpOverload):
        overloads.append(op)
    else:
        if not isinstance(op, OpOverloadPacket):
            raise AssertionError("op must be instance of OpOverloadPacket.")
        for ol in op.overloads():
            overloads.append(getattr(op, ol))

    for op_overload in overloads:
        if op_overload in npu_meta_table:
            raise RuntimeError(f"duplicate registrations for npu_meta_table {op_overload}")
        npu_meta_table[op_overload] = fn
        if avoid_fallback_flag:
            avoid_make_fallback_table.append(op_overload)


def register_meta_npu(op, avoid_fallback_flag=False):
    def meta_decorator(fn: Callable):
        _add_op_to_meta_table(op, fn, avoid_fallback_flag)
        return fn

    return meta_decorator


def register_break_fn(meta_class, op_name):
    op_name = op_name.upper()

    def decorator(func: Callable):
        break_fn_table[op_name] = func
        break_mapping_table[op_name] = meta_class
        return func

    return decorator



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


raw_batch_norm_func = decomposition_table[aten._native_batch_norm_legit_no_training.default]


@register_meta_npu(aten._native_batch_norm_legit_no_training.default)
def meta__native_batch_norm_legit_no_training(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    output, save_mean, save_rstd = raw_batch_norm_func(input, weight, bias, running_mean, 
                                                       running_var, momentum, eps)
    # npu's impl follows cpu's, and npu's [save_mean, save_rstd] shape is different from gpu's and
    # meta func. So in npu_backend, save_mean and save_rstd will be always emtpy tensor.
    save_mean = input.new_zeros((0,))
    save_rstd = input.new_zeros((0,))
    return output, save_mean, save_rstd


@register_meta_npu(aten.view_as_complex.default, True)
def meta_view_as_complex(self: Tensor):
    out_shape = _broadcast_shapes(self.shape)
    return self.new_empty(out_shape, dtype=corresponding_complex_dtype(self.dtype))


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
        if not isinstance(op_overload, OpOverload):
            raise AssertionError("op_overload must be instance of OpOverload.")
        if op_overload not in avoid_make_fallback_table:
            decomposition_table[op_overload] = fn
        op_overload.py_kernels.pop(DispatchKey.Meta, None)
        op_overload.py_impl(DispatchKey.Meta)(fn)

    patch_torch_decomp_decompositions()


@register_break_fn('BINARY_SUBSCR', 'SLICE')
def break_fn_slice(self, inst):
    """
    break op slice/slice_backward. 
    For example:
        x = x[:10, :20, :, :] # slice
    """
    if isinstance(self.stack[1], TupleVariable):
        for var in self.stack[1].as_proxy():
            if isinstance(var, slice):
                raise Unsupported(f"Unsupported slice op !")
            
            
@register_break_fn('STORE_SUBSCR', 'SETITEM')
def break_fn_setitem(self, inst):
    """
    break ops like index_put, slice_scatter, etc. 
    For example:
        x[:10] = 1 # slice_scatter
        x[[1, 3, 4]] = 1 # index_put
    """
    raise Unsupported(f"Unsupported setitem op !")
            
            
@register_break_fn('CALL_FUNCTION', 'NN.CONV3D')
def break_fn_conv3d(self, inst):
    """
    break module nn.Conv3d. 
    For example:
        torch.nn.Conv3d
    """
    if isinstance(self.stack[0], NNModuleVariable):
        if self.stack[0].module_type == torch.nn.Conv3d:
            raise Unsupported(f"Unsupported conv3d module !")


@register_break_fn('CALL_FUNCTION', 'NN.LINEAR')
def break_fn_linear(self, inst):
    """
    break module nn.Linear.
    For example:
        torch.nn.Linear
    """
    for stack in self.stack:
        if isinstance(stack, NNModuleVariable) and stack.module_type == torch.nn.Linear:
            raise Unsupported(f"Break graph for linear module !")


def add_break_graph(op_table):
    """
    Initially support:
        subscr: slice/slice_backward
        call_function: nn.Conv3d
        store_subscr: scatter_add/index_put
    How to customize your own graph breaking function:
        1、Figure out which function to be patched.
        2、Register your own break_fn.
        3、Implement new patch functions.
    """
    @break_graph_if_unsupported(push=1)
    def binary_subscr(self, inst):
        if 'BINARY_SUBSCR' in op_table:
            for op_fn in op_table['BINARY_SUBSCR']:
                op_fn(self, inst)
        stack_op(operator.getitem)(self, inst)
        
    @break_graph_if_unsupported(push=1)
    def call_function(self, inst):
        if 'CALL_FUNCTION' in op_table:
            for op_fn in op_table['CALL_FUNCTION']:
                op_fn(self, inst)
        args = self.popn(inst.argval)
        fn = self.pop()
        self.call_function(fn, args, {})
        
    @break_graph_if_unsupported(push=0)
    def store_subscr(self, inst):
        if 'STORE_SUBSCR' in op_table:
            for op_fn in op_table['STORE_SUBSCR']:
                op_fn(self, inst)
        val, obj, key = self.popn(3)
        result = obj.call_method(self, "__setitem__", [key, val], {})
        # no result is pushed, so need to lift the guards to global
        self.output.guards.update(result.guards)
        
    InstructionTranslatorBase.BINARY_SUBSCR = binary_subscr
    InstructionTranslatorBase.STORE_SUBSCR = store_subscr
    InstructionTranslatorBase.CALL_FUNCTION = call_function
    

def npu_patch_break_graph():
    """
    Automatically break graph through op or module. Initially support slice/slice_backward, 
    setitem(index_put, slice_scatter), nn.Conv3d by setting environment variable BREAK_GRAPH_OP_LIST.
    Usage:
        If you want to break down only `slice` and `nn.Conv3d`, you can achieve it by setting 
        environment variable BREAK_GRAPH_OP_LIST="SLICE,NN.CONV3D" 
    Notice: 
        Valid ops or modules specified by BREAK_GRAPH_OP_LIST is split by ','
    """
    break_graph_op_list = os.getenv('BREAK_GRAPH_OP_LIST', False)
    if not break_graph_op_list:
        return
    
    def add_value_to_dict(my_dict, key, value):
        if key not in my_dict:
            my_dict[key] = [value]
        else:
            my_dict[key].append(value)
    
    op_table = {}
    for op_name in break_graph_op_list.split(','):
        op_name = op_name.upper()
        if op_name in break_mapping_table:
            patch_fn = break_mapping_table.get(op_name)
            op_fn = break_fn_table.get(op_name)
            add_value_to_dict(op_table, patch_fn, op_fn)
        else:
            raise RuntimeError(
                f"Setting BREAK_GRAPH_OP_LIST ERROR: "
                f"Invalid break op `{op_name}`, please register "
                f"break fn for `{op_name}` first. The available options "
                f"are: {list(break_mapping_table.keys())}")
            
    add_break_graph(op_table)


def register_matmul_backward_decomp():
    '''
    Torch_npu currently dispatch linear to matmul and matmul_backward
    instead of mm and mm_backwrd. This will lead to some bug in npu_backend
    and the decomposition can fix it.
    '''
    @aten.to.other.py_impl(DispatchKey.AutogradPrivateUse1)
    def to_other_decomposition(self, other, non_blocking=False, copy=False, memory_format=None):
        output = self.to(other.dtype)
        return output

    @aten.type_as.default.py_impl(DispatchKey.AutogradPrivateUse1) 
    def type_as_decomposition(self, other):
        output = self.to(other.dtype)
        return output
    
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
        
        def matmul_backward_1d_1d():
            nonlocal grad_self, grad_other
            grad_self = other.mul(grad) if mask[0] else grad_self
            grad_other = self.mul(grad) if mask[1] else grad_other
            return grad_self, grad_other
        
        def matmul_backward_2d_1d():
            nonlocal grad_self, grad_other
            grad_self = grad.unsqueeze(1).mm(other.unsqueeze(0)) if mask[0] else grad_self
            grad_other = self.transpose(-1, -2).mm(grad.unsqueeze(1)).squeeze_(1) if mask[1] else grad_other
            return grad_self, grad_other
        
        def matmul_backward_1d_2d():
            nonlocal grad_self, grad_other
            grad_self = grad.unsqueeze(0).mm(other.transpose(-1, -2)).squeeze_(0) if mask[0] else grad_self
            grad_other = self.unsqueeze(1).mm(grad.unsqueeze(0)) if mask[1] else grad_other
            return grad_self, grad_other
        
        def matmul_backward_nd_lt3d():
            nonlocal grad_self, grad_other
            view_size = 1 if dim_other == 1 else size_grad[-1]
            unfolded_grad = (grad.unsqueeze(-1) if dim_other == 1 else grad).contiguous().view(-1, view_size)
            if mask[0]:
                unfolded_other = other.unsqueeze(0) if dim_other == 1 else other.transpose(-1, -2)
                grad_self = unfolded_grad.mm(unfolded_other).view(size_self)

            if mask[1]:
                # create a 2D-matrix from self
                unfolded_self = self.contiguous().view(-1, size_self[-1])
                grad_other = unfolded_self.transpose(-1, -2).mm(unfolded_grad).view(size_other)
            return grad_self, grad_other
        
        def matmul_backward_lt3d_nd():
            nonlocal grad_self, grad_other
            view_size = 1 if dim_self == 1 else size_grad[-2]
            unfolded_grad_t = grad.view(-1, view_size) if dim_self == 1 else \
                                                            grad.transpose(-1, -2).contiguous().view(-1, view_size)
            if mask[0]:
                # create a 2D-matrix from other
                unfolded_other_t = \
                    other.transpose(-1, -2).contiguous().view(-1, size_other[-2]).transpose(-1, -2)
                grad_self = unfolded_other_t.mm(unfolded_grad_t).transpose(-1, -2).view(size_self)

            if mask[1]:
                size_other_t = list(size_other[:-2])
                size_other_t.extend([size_other[dim_other - 1], size_other[dim_other - 2]])
                unfolded_self = self.unsqueeze(0) if dim_self == 1 else self
                grad_other = unfolded_grad_t.mm(unfolded_self).view(size_other_t).transpose(-1, -2)
            return grad_self, grad_other
                
        if dim_self == 1 and dim_other == 1:
            grad_self, grad_other = matmul_backward_1d_1d()
        elif dim_self == 2 and dim_other == 1:
            grad_self, grad_other = matmul_backward_2d_1d()
        elif dim_self == 1 and dim_other == 2:
            grad_self, grad_other = matmul_backward_1d_2d()
        elif dim_self >= 3 and (dim_other == 1 or dim_other == 2):
            # create a 2D-matrix from grad
            grad_self, grad_other = matmul_backward_nd_lt3d()
        elif (dim_self == 1 or dim_self == 2) and dim_other >= 3:
            # create a 2D-matrix from grad
            grad_self, grad_other = matmul_backward_lt3d_nd()
        else:
            grad_self = torch.matmul(grad, other.transpose(-1, -2)) if mask[0] else grad_self
            grad_other = torch.matmul(self.transpose(-1, -2), grad) if mask[1] else grad_other

        return grad_self, grad_other
    

def npu_patch_fx_pass(decompositions):
    """
    Notes:
        1. replace joint_graph.lazy_init with custom lazy_init
          fx passes are enabled by lazy_init. This patch disable inductor's fx passes 
          (which is useless and affects npu fx graph), and enable custom npu fx passes.  
        2. replace inductor's decompositions with user-defined decompositions.
          Original joint_graph_pass will decomposite fx graph through inductor's decompositions,
          which is not compatible with user defined decompositions, and cause the failure of 
          pattern match. 
          Here we clear joint_graph.patterns to avoid extra patterns' effects.
    """
    from torch._inductor.fx_passes import joint_graph
    from torch._inductor import decomposition
    from .npu_fx_passes.joint_graph import lazy_init

    joint_graph.patterns.clear()
    decomposition.decompositions.clear()
    decompositions_for_rng.extra_random_decomps.clear()
    decomposition.decompositions.update(decompositions)
    
    joint_graph.lazy_init = lazy_init


def npu_patch_register_fast_op_impl():
    # Fix div dtype infer bug in dynamo when inputs have both IntTensor and SymInt/int.
    try:
        from torch._subclasses.fake_tensor import get_fast_op_impls, register_fast_op_impl, \
                                                FAST_OP_IMPLEMENTATIONS, make_fast_binary_impl
    except ImportError:
        # In torch version >= 2.3, apis about fast_op_impl are removed and this patch is skipped.
        # So in torch version >= 2.3, div's dtype may be not calculated correctly.
        return

    def is_int(dtype):
        if dtype in [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]:
            return True
        return False

    src_impl = make_fast_binary_impl(refs_div)

    def make_div_binary_impl(mode, *args, **kwargs):
        result = src_impl(mode, *args, **kwargs)
        operands = args
        tensor_all_int = True
        has_int_or_symint = False
        for op in operands:
            if isinstance(op, (torch.SymInt, int)):
                has_int_or_symint = True
            if isinstance(op, Tensor) and not is_int(op.dtype):
                tensor_all_int = False
        if tensor_all_int and has_int_or_symint:
            result = result.to(torch.float)
        return result

    src_get_fast_op_impls = get_fast_op_impls

    @lru_cache(None)
    def new_get_fast_op_impls():
        src_get_fast_op_impls()
        register_fast_op_impl(aten.div.Tensor)(make_div_binary_impl)
        return FAST_OP_IMPLEMENTATIONS
    _subclasses_fake_tensor.get_fast_op_impls = new_get_fast_op_impls


@run_once
def add_npu_patch(decompositions, compiler_config):
    adjust_implicit_decomposition()
    adjust_traceable_collective_remaps()
    register_matmul_backward_decomp()
    npu_patch_meta()
    npu_patch_break_graph()
    npu_patch_register_fast_op_impl()
    if compiler_config.experimental_config.npu_fx_pass:
        npu_patch_fx_pass(decompositions)


def get_npu_default_decompositions():
    default_decompositions = {}
    from torchair._ge_concrete_graph.ge_converter.experimental.hcom_allgather import allgather_decomposition
    default_decompositions.update({torch.ops.npu_define.allgather.default: allgather_decomposition})
    if torch.__version__ >= "2.3.1":
        from torchair._ge_concrete_graph.ge_converter.c10d_functional.c10d_functional import \
            decomp_c10d_functional_all_to_all_single
        default_decompositions.update(
            {torch.ops._c10d_functional.all_to_all_single.default: decomp_c10d_functional_all_to_all_single})
    return default_decompositions
