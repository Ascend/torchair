import os
from typing import List

import torch
import numpy as np
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, ControlTensor, DataType
from torchair.ge._ge_graph import dont_prune_me
from torchair.ge._ge_graph import get_default_ge_graph
from torchair.ge import attr
from torchair._ge_concrete_graph import ge_apis as ge
from ._lib import lib

lib.define(
    """
    print(Tensor[] tensors, Dimname msg_format, Dimname placeholder, int summarize_num) -> Tensor[]
    """
)


def summarize_tensor(tensor: torch.Tensor, summarize_num) -> str:
    def _value(v):
        return f"{v:.0f}" if v.is_integer() else f"{v}"

    def _summarize_array(array: np.ndarray, nums):
        if not isinstance(array, np.ndarray):
            return _value(array)

        if array.ndim == 1:
            if array.shape[0] <= 2 * nums:
                return f"[{' '.join([_value(a) for a in array])}]"
            first = ' '.join([_value(a) for a in array[:nums]])
            last = ' '.join([_value(a) for a in array[-nums:]])
            return f"[{first} ... {last}]"

        if array.shape[0] <= 2 * nums:
            sub_arrays_str = '\n'.join([_summarize_array(a, nums) for a in array])
            return f"[{sub_arrays_str}]"

        sub_arrays = [_summarize_array(a, nums) for a in array[:nums]]
        sub_arrays.append('...')
        sub_arrays.extend([_summarize_array(a, nums) for a in array[-nums:]])
        sub_arrays_str = '\n'.join(sub_arrays)
        return f"[{sub_arrays_str}]"

    if tensor.dtype == torch.bfloat16:
        tensor = tensor.detach().float()
    return _summarize_array(tensor.detach().cpu().numpy(), summarize_num)


@torch.library.impl(lib, "print", "Meta")
def kernel(
        tensors: List[torch.Tensor],
        msg_format: str,
        placeholder: str,
        summarize_num: int
) -> List[torch.Tensor]:
    if len(tensors) == 0:
        raise RuntimeError("print op requires at least one tensor")
    return tensors


def _print_tensors(tensors: List[torch.Tensor],
                   msg_format: str,
                   placeholder: str,
                   summarize_num: int):
    for t in tensors:
        tensor_str = summarize_tensor(t.detach(), summarize_num)
        msg_format = msg_format.replace(placeholder, tensor_str, 1)
    print(msg_format)


@torch.library.impl(lib, "print", "CPU")
def kernel(
        tensors: List[torch.Tensor],
        msg_format: str,
        placeholder: str,
        summarize_num: int
) -> List[torch.Tensor]:
    _print_tensors(tensors, msg_format, placeholder, summarize_num)
    return tensors


@torch.library.impl(lib, "print", "PrivateUse1")
def kernel(
        tensors: List[torch.Tensor],
        msg_format: str,
        placeholder: str,
        summarize_num: int
) -> List[torch.Tensor]:
    _print_tensors(tensors, msg_format, placeholder, summarize_num)
    return tensors


def _npu_print(*args, summarize_num):
    joint_str = " ".join([arg if isinstance(arg, str) else "[]" for arg in args])
    i = 0
    placeholder = '{}'
    while placeholder in joint_str:
        placeholder = f'{{{i}}}'
        i += 1
    msg_format = " ".join([arg if isinstance(arg, str) else placeholder for arg in args])
    tensors = [arg for arg in args if not isinstance(arg, str)]
    return torch.ops.air.print(tensors, msg_format, placeholder, summarize_num)


@register_fx_node_ge_converter(torch.ops.air.print.default)
def convert_print(tensors: List[Tensor],
                  msg_format: str,
                  placeholder: str,
                  summarize_num: int,
                  meta_outputs: List[TensorSpec] = None):
    printable_tensors = [ge.Cast(t, dst_type=DataType.DT_FLOAT) if t.dtype == DataType.DT_BF16 else t for t in tensors]
    msg = ge.StringFormat(printable_tensors, template=msg_format, placeholder=placeholder, summarize=summarize_num)
    ge.PrintV2(msg)
    print_op = get_default_ge_graph().op[-1]
    attr.Str('extend').merge_to(print_op.attr['_kernel'])
    outputs = ControlTensor(print_op)
    dont_prune_me(outputs)
    return [ge.Identity(x, dependencies=[outputs]) for x in tensors]
