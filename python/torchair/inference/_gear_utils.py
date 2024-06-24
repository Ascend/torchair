import itertools
import functools
from typing import Any, Dict, List, Tuple, Union, Callable

import torch
from torch.fx.experimental.symbolic_shapes import guard_int, guard_bool
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge_concrete_graph.ge_graph import _TensorInput, _DiscontiguousTensorInput, is_sym
from torchair.core.utils import logger


def set_dim_gears(t: torch.Tensor, dim_gears: Dict[int, List[int]]):
    def check_len(gears):
        return len(gears) < 2 or len(gears) > 100
    def check_range(gears):
        return max(gears) > 2048 or min(gears) < 1
    def check_int_list(gears):
        return any(not isinstance(gear, int) for gear in gears)

    if any(check_len(gears) or check_int_list(gears) or check_range(gears) for gears in dim_gears.values()):
        raise AssertionError(f'The gears value list size is at least 2 and not over 100, '
                             f'and value must in range [1, 2048], type is int, '
                             f'but config gears is {dim_gears}')

    for key in dim_gears.keys():
        torch._dynamo.mark_dynamic(t, key)

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

    duplicated_dynamic_dims = list(set(dynamic_dims))
    if len(duplicated_dynamic_dims) > 100:
        raise AssertionError(f'The total number of gears set cannot exceed 100, '
                             f'and the current number of gears is: {len(duplicated_dynamic_dims)}')

    option_dynamic_dims = ';'.join([','.join(map(str, sublist)) for sublist in duplicated_dynamic_dims])
    ge_option["ge.dynamicDims"] = option_dynamic_dims

    return ge_option
