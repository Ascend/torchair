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
    if any(len(gears) < 2 for gears in dim_gears.values()):
        raise AssertionError(f'The gears size at least 2, but config gears is {dim_gears}')

    for key in dim_gears.keys():
        torch._dynamo.mark_dynamic(t, key)

    setattr(t, "dim_gears", dim_gears)


def get_dim_gears(t: torch.Tensor):
    return getattr(t, "dim_gears", None)


def _guard_dynamic_dim(dim, dynamic_gears):
    gear_guard = functools.reduce(lambda x, y: x | y, [dim == gear for gear in dynamic_gears])
    if not bool(gear_guard):
        raise AssertionError(f'The index {dim} of the current tensor shape '
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
        for dim_index, gear in dim_gears.items():
            if data_shape[dim_index] != -1:
                raise AssertionError(f"Set dim gear index shape must generalize -1.")
            gear_list.append(tuple(gear))

    ge_option['ge.inputShape'] = str_inputshape.rstrip(";")

    if len(gear_list) == 0:
        raise AssertionError(f"At least one dimension should have levels.")

    if config == "zip":
        if all(len(sublist) == len(gear_list[0]) for sublist in gear_list):
            dynamic_dims = ';'.join([','.join(map(str, sublist)) for sublist in list(zip(*gear_list))])
        else:
            raise AssertionError("when dynamic_gears_merge_policy is zip, input gears len must same.")
    elif config == "product":
        dynamic_dims = ';'.join([','.join(map(str, sublist)) for sublist in list(itertools.product(*gear_list))])
    else:
        raise AssertionError("dynamic_gears_merge_policy only support zip and product.")

    ge_option["ge.dynamicDims"] = dynamic_dims

    return ge_option
