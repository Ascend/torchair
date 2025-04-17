from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)
import sys

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support


def check_if_skip(start, end, step):
    if isinstance(start, Tensor) or isinstance(end, Tensor) or isinstance(step, Tensor):
        return False
    if start == 0 and end == sys.maxsize and step == 1:
        return True
    return False


def static_gen(input_sizes, dim, start, end, dtype):
    # Generate correct tensor for concating grad.
    new_sizes = input_sizes.copy()
    real_dim = end - start
    real_dim = max(real_dim, 0)
    new_sizes[dim] = real_dim
    return ge.Fill(new_sizes, ge.Cast(0., dst_type=dtype))


def static_concat(grad_output, input_sizes, dim, start, end, meta_outputs):
    length = input_sizes[dim]
    result = []

    if start >= length:
        start = length
    else:
        start = start % length

    if end >= length:
        end = length
    else:
        end = end % length

    start = min(start, end)
    zeros1 = static_gen(input_sizes, dim, 0, start, meta_outputs.dtype)
    result.append(zeros1)

    result.append(grad_output)

    zeros2 = static_gen(input_sizes, dim, end, length, meta_outputs.dtype)
    result.append(zeros2)

    grad_input = ge.ConcatV2(result, concat_dim=dim, N=len(result))
    return grad_input


def dynamic_concat(grad_output, input_sizes, dim, start, end, meta_outputs):
    length = ge.Gather(input_sizes, dim)
    result = []

    start = ge.SelectV2(ge.GreaterEqual(start, length), length, ge.FloorMod(start, length))
    end = ge.SelectV2(ge.GreaterEqual(end, length), length, ge.FloorMod(end, length))

    start = ge.Minimum(start, end)
    zeros1 = dynamic_gen(grad_output, input_sizes, dim, 0, start, meta_outputs.dtype)
    result.append(zeros1)

    result.append(grad_output)

    zeros2 = dynamic_gen(grad_output, input_sizes, dim, end, length, meta_outputs.dtype)
    result.append(zeros2)

    grad_input = ge.ConcatV2(result, concat_dim=dim, N=len(result))
    return grad_input


def dynamic_gen(grad_output, input_sizes, dim, start, end, dtype):
    # Generate correct tensor for concating grad.
    real_dim = ge.Sub(end, start)
    real_dim = ge.Maximum(real_dim, 0)
    new_sizes = []
    for i in range(grad_output.rank):
        if i == dim:
            new_sizes.append(real_dim)
        else:
            new_sizes.append(ge.Gather(input_sizes, i))
    new_sizes = ge.Pack(new_sizes, N=grad_output.rank, axis=0)
    return ge.Fill(new_sizes, ge.Cast(0., dst_type=dtype))


@declare_supported([
    Support(F32(3, 4), [3, 8], 1, 0, 8, 2),
    Support(F16(3, 4), [3, 8], 1, 0, 10, 2),
])
@register_fx_node_ge_converter(torch.ops.aten.slice_backward.default)
def conveter_aten_slice_backward_default(
    grad_output: Tensor,
    input_sizes: Union[List[int], Tensor],
    dim: int,
    start: Union[int, Tensor],
    end: Union[int, Tensor],
    step: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::slice_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step) -> Tensor"""
    if check_if_skip(start, end, step):
        return grad_output

    if isinstance(step, int) and step == 1:
        # If step is 1, optimize slice_backward by ConcatV2.
        if isinstance(start, int) and isinstance(end, int) and isinstance(input_sizes, list):
            return static_concat(grad_output, input_sizes, dim, start, end, meta_outputs)
        else:
            return dynamic_concat(grad_output, input_sizes, dim, start, end, meta_outputs)
        
    src_input_sizes = input_sizes
    src_end = end

    if isinstance(input_sizes, list) and isinstance(end, int):
        end = input_sizes[dim] if end == sys.maxsize or input_sizes[dim] < end else end
        end = input_sizes[dim] + end if end < 0 else end
        
    input_sizes, start, limit, delta = dtype_promote(input_sizes, start, end, step, target_dtype=torch_type_to_ge_type(torch.int32))
    
    grad_input = ge.Fill(input_sizes, ge.Const(0., dtype=grad_output.dtype))
    dims_to_expand = [i for i in range(grad_output.rank)]
    dims_to_expand.remove(dim)

    if isinstance(src_input_sizes, Tensor) or isinstance(src_end, Tensor):
        limit = ge.Minimum(ge.Gather(input_sizes, dim), limit)

    idx = ge.Range(start, limit, delta)

    if dims_to_expand:
        idx_unsqueezed = ge.Unsqueeze(idx, axes=dims_to_expand)
        idx_expanded = ge.Expand(idx_unsqueezed, ge.Shape(grad_output))
    else:
        idx_expanded = idx

    return ge.ScatterElements(grad_input, idx_expanded, grad_output, axis=dim)    


@register_fx_node_ge_converter(torch.ops.aten.slice_backward.out)
def conveter_aten_slice_backward_out(
    grad_output: Tensor,
    input_sizes: Union[List[int], Tensor],
    dim: int,
    start: Union[int, Tensor],
    end: Union[int, Tensor],
    step: Union[int, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::slice_backward.out(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.slice_backward.out ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")