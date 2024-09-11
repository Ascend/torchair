from typing import List

import torch
from torch.fx.node import has_side_effect
import numpy as np
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, ControlTensor, DataType
from torchair.ge._ge_graph import dont_prune_me
from torchair.ge._ge_graph import get_default_ge_graph
from torchair.ge import attr
from torchair._ge_concrete_graph import ge_apis as ge
from torchair.core._backend import init_device_stdout_channel
from ._lib import lib

lib.define(
    """
    print(Tensor[] tensors, str msg_format, str placeholder, int summarize_size) -> None
    """
)
has_side_effect(torch.ops.air.print.default)


def summarize_tensor(tensor: torch.Tensor, summarize_size) -> str:
    def _value(v):
        v = float(v)
        return f"{v:.0f}" if v.is_integer() else f"{v}"

    def _summarize_array(array, nums):
        if not isinstance(array, np.ndarray):
            return _value(array)

        if array.ndim == 0:
            return _value(array.item())

        if array.ndim == 1:
            if nums < 0 or array.shape[0] <= 2 * nums:
                return f"[{' '.join([_value(a) for a in array])}]"
            first = ' '.join([_value(a) for a in array[:nums]])
            last = ' '.join([_value(a) for a in array[array.shape[0] - nums:]])
            return f"[{first} ... {last}]"

        if nums < 0 or array.shape[0] <= 2 * nums:
            sub_arrays_str = '\n '.join([_summarize_array(a, nums) for a in array])
            return f"[{sub_arrays_str}]"

        sub_arrays = [_summarize_array(a, nums) for a in array[:nums]]
        sub_arrays.append('...')
        sub_arrays.extend([_summarize_array(a, nums) for a in array[array.shape[0] - nums:]])
        sub_arrays_str = '\n '.join(sub_arrays)
        return f"[{sub_arrays_str}]"

    if tensor.dtype == torch.bfloat16:
        tensor = tensor.detach().float()
    if summarize_size < -1 or summarize_size == 0:
        raise ValueError(f"summarize_size must be positive or -1 for all elements, got {summarize_size}")
    return _summarize_array(tensor.detach().cpu().numpy(), summarize_size)


@torch.library.impl(lib, "print", "Meta")
def kernel_meta(
        tensors: List[torch.Tensor],
        msg_format: str,
        placeholder: str,
        summarize_size: int
):
    return None


def kernel_impl(
        tensors: List[torch.Tensor],
        msg_format: str,
        placeholder: str,
        summarize_size: int
):
    for t in tensors:
        tensor_str = summarize_tensor(t.detach(), summarize_size)
        msg_format = msg_format.replace(placeholder, tensor_str, 1)
    print(msg_format, flush=True)


torch.library.impl(lib, "print", "CPU")(kernel_impl)
torch.library.impl(lib, "print", "PrivateUse1")(kernel_impl)


def _npu_print(*args, summarize_size):
    for i, arg in enumerate(args):
        if not isinstance(arg, (torch.Tensor, str, bool, float, int)):
            raise TypeError(f"position arguments of npu_print() must be torch.Tensor or "
                            f"basic type in (str, bool, float, int), but got '{type(arg).__name__}' at position {i}")
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    if len(tensors) == 0:
        raise ValueError("npu_print() requires at least one tensor input")
    if not isinstance(summarize_size, int):
        raise TypeError(f"summarize_size must be int, got {type(summarize_size).__name__}")
    if summarize_size < -1 or summarize_size == 0:
        raise ValueError(f"summarize_size must be positive or -1 for all elements, got {summarize_size}")

    joint_str = " ".join(["[]" if isinstance(arg, torch.Tensor) else str(arg) for arg in args])
    i = 0
    placeholder = '{}'
    while placeholder in joint_str:
        placeholder = f'{{{i}}}'
        i += 1
    msg_format = " ".join([placeholder if isinstance(arg, torch.Tensor) else str(arg) for arg in args])
    torch.ops.air.print(tensors, msg_format, placeholder, summarize_size)


@register_fx_node_ge_converter(torch.ops.air.print.default)
def convert_print(tensors: List[Tensor],
                  msg_format: str,
                  placeholder: str,
                  summarize_size: int,
                  meta_outputs: List[TensorSpec] = None):
    printable_tensors = [ge.Cast(t, dst_type=DataType.DT_FLOAT) if t.dtype == DataType.DT_BF16 else t for t in tensors]
    msg = ge.StringFormat(printable_tensors, template=msg_format, placeholder=placeholder, summarize=summarize_size)
    ge.PrintV2(msg)
    print_op = get_default_ge_graph().op[-1]
    attr.Str('extend').merge_to(print_op.attr['_kernel'])
