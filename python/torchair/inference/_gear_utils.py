import itertools
import functools
from typing import Any, Dict, List, Tuple, Union, Callable

import torch
from torch.fx.experimental.symbolic_shapes import guard_int, guard_bool
from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge._ge_graph import _TensorInput, _DiscontiguousTensorInput, is_sym
from torchair.core.utils import logger


def set_dim_gears(t: torch.Tensor, dim_gears: Dict[int, Union[List[int], Tuple[int, ...]]]):
    if get_dim_gears(t) is not None and get_dim_gears(t) != dim_gears:
        raise AssertionError(
            f"Tensor {t} already has set dim gears, and it is not supported to set it again.")
    for dim_index, gears in dim_gears.items():
        if not isinstance(dim_index, int):
            raise AssertionError(
                f"Dim index in dim_gears must be an integer, but got {type(dim_index)}.")
        if dim_index < 0 or dim_index >= len(t.shape):
            raise AssertionError(
                f"Dim index in dim_gears must be in range [0, {len(t.shape) - 1}], but got {dim_index}.")
        if not isinstance(gears, (list, tuple)):
            raise AssertionError(
                f"Gears for dim index {dim_index} in dim_gears must be a list or tuple, but got {type(gears)}.")
        for index, gear in enumerate(gears):
            if not isinstance(gear, int):
                raise AssertionError(f"Element at index {index} of value for dim index {dim_index} in "
                                     f"dim_gears must be an integer, but got {type(gear)}.")
        if not 2 <= len(gears) <= 100:
            raise AssertionError(f"Length of gears for dim index {dim_index} in dim_gears must be in "
                                 f"range [2, 100], but got {len(gears)}.")

        torch._dynamo.mark_dynamic(t, dim_index)

    setattr(t, "dim_gears", dim_gears)


def get_dim_gears(t: torch.Tensor):
    return getattr(t, "dim_gears", None)


def _guard_dynamic_dim(dim, dynamic_gears):
    sorted_gears = sorted(list(set(dynamic_gears)))
    gear_guard = (dim >= sorted_gears[0]) & (dim <= sorted_gears[-1])
    if not bool(gear_guard):
        raise AssertionError(f'The index {int(dim)} of the current tensor shape '
                             f'is not within the range {dynamic_gears}')
    guard_bool(gear_guard)

    for i, gear in enumerate(sorted_gears[:-1]):
        left = gear
        right = sorted_gears[i + 1]
        gear_guard = (dim <= left) | (dim >= right)
        if not bool(gear_guard):
            raise AssertionError(f'The index {int(dim)} of the current tensor shape '
                                 f'is not within the range {dynamic_gears}')
        guard_bool(gear_guard)


def guard_gears_shape(example_inputs):
    if not any(isinstance(t, torch.Tensor) and get_dim_gears(t) is not None for t in example_inputs):
        return

    for t in example_inputs:
        if not isinstance(t, torch.Tensor):
            continue
        dim_gear = get_dim_gears(t)
        if dim_gear is None or len(dim_gear) == 0:
            [guard_int(dim) for dim in t.size()]
            continue
        for i, dim in enumerate(t.size()):
            if i not in dim_gear.keys():
                guard_int(dim)
                continue
            _guard_dynamic_dim(dim, dim_gear[i])


def generate_dynamic_dims_option(named_inputs_info, config):
    if not any(len(t.dim_gears) != 0 for t in named_inputs_info.values()):
        return {}
    ge_option = {"ge.inputShape" : None, "ge.dynamicDims": None, "ge.dynamicNodeType": "1"}
    sorted_named_inputs = {k: v for k, v in sorted(named_inputs_info.items(), key=lambda x: x[0])}
    str_inputshape = ''
    gear_list = []
    for op_name, ge_input_info in sorted_named_inputs.items():
        data_shape = ge_input_info.shape
        dim_gears = ge_input_info.dim_gears
        str_inputshape += f"{op_name}:{','.join(map(str, data_shape))};"
        if len(dim_gears) == 0:
            continue
        sorted_dim_gears = dict(sorted(dim_gears.items(), key=lambda x: x[0]))
        for dim_index, gear in sorted_dim_gears.items():
            if data_shape[dim_index] == -1:
                gear_list.append(tuple(gear))

    ge_option['ge.inputShape'] = str_inputshape.rstrip(";")

    if len(gear_list) == 0:
        raise AssertionError(f"At least one dimension should have levels.")

    if config == "zip":
        if all(len(sublist) == len(gear_list[0]) for sublist in gear_list):
            dynamic_dims = list(zip(*gear_list))
        else:
            raise AssertionError("when dynamic_gears_merge_policy is zip, input gears len must same.")
    elif config == "product":
        dynamic_dims = list(itertools.product(*gear_list))
    else:
        raise AssertionError("dynamic_gears_merge_policy only support zip and product.")

    unique_dynamic_dims = list(set(dynamic_dims))
    if len(unique_dynamic_dims) > 100:
        raise AssertionError(f'The total number of gears set cannot exceed 100, '
                             f'and the current number of gears is: {len(unique_dynamic_dims)}')

    option_dynamic_dims = ';'.join([','.join(map(str, sublist)) for sublist in unique_dynamic_dims])
    ge_option["ge.dynamicDims"] = option_dynamic_dims

    return ge_option
