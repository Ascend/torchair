"""Functions used for NPU device"""

import os
import threading
import logging
import atexit
import sys
from collections import defaultdict

from typing import Dict

import torch
from torchair._utils.error_code import pretty_error_msg

from . import _torchair


def stupid_repeat(word, times):
    return _torchair.StupidRepeat(word, times)


def _try_get_torch_npu_device():
    if 'torch_npu' not in sys.modules:
        return None

    from . import _npu_graph_executor

    torch_npu_module = sys.modules['torch_npu']
    return torch_npu_module.npu.current_device()


def _get_global_op_compile_config():
    op_compile_config = dict()
    if 'torch_npu' not in sys.modules:
        op_compile_config['ge.exec.allow_hf32'] = '10'  # enable conv hf32 as default
        op_compile_config['ge.deterministic'] = '1'
        return op_compile_config

    torch_npu_module = sys.modules['torch_npu']
    mm_hf32 = '1' if torch_npu_module.npu.matmul.allow_hf32 else '0'
    conv_hf32 = '1' if torch_npu_module.npu.conv.allow_hf32 else '0'
    op_compile_config['ge.exec.allow_hf32'] = conv_hf32 + mm_hf32
    op_compile_config['ge.deterministic'] = '1' if torch.are_deterministic_algorithms_enabled() else '0'
    return op_compile_config


# only init once when session Initialize
_GLOBAL_COMPILE_OPTION = None


# GE is initialized only once. Therefore, if set different initialization options twice,
# it will raise ValueError
def _try_get_global_init_compile_option(global_options: Dict = None):
    def is_subset(subset, superset):
        for key, value in subset.items():
            if key not in superset or superset[key] != value:
                raise ValueError(
                    'Unsupport different initialization options twice, new option value [{}] is different from init '
                    'option value [{}] while option key is [{}], please check your compile config when use '
                    'torch.compile()'.format(value, superset[key], key))
        return True

    global _GLOBAL_COMPILE_OPTION
    if _GLOBAL_COMPILE_OPTION is None:
        _GLOBAL_COMPILE_OPTION = global_options if global_options else {}
    else:
        if not (global_options is None or is_subset(global_options, _GLOBAL_COMPILE_OPTION)):
            raise AssertionError

    return _GLOBAL_COMPILE_OPTION


@pretty_error_msg
def initialize_graph_engine(global_compile_options: Dict = None):
    options: Dict[str, str] = {}
    options.update(_try_get_global_init_compile_option(global_compile_options))
    torch_npu_device = _try_get_torch_npu_device()
    if torch_npu_device is not None:
        options['ge.exec.deviceId'] = str(torch_npu_device)
        options['ge_run_with_torch_npu'] = "1"
    else:
        options['ge.exec.deviceId'] = os.getenv('ASCEND_DEVICE_ID', '0')
        options['ge_run_with_torch_npu'] = "0"

    options.update(_get_global_op_compile_config())
    _torchair.InitializeGraphEngine(options)


@pretty_error_msg
def finalize_graph_engine():
    _torchair.FinalizeGraphEngine()


atexit.register(finalize_graph_engine)


class TorchNpuGraph(_torchair.TorchNpuGraphBase):
    def __init__(self, name=""):
        super(TorchNpuGraph, self).__init__(str(name))

    @pretty_error_msg
    def load(self, ge_graph, options=None):
        """Load the graph"""
        options = {} if options is None else options
        input_placements = ge_graph.attr["_input_placements"].list.i
        output_dtypes = ge_graph.attr["_output_dtypes"].list.i
        executor_type = ge_graph.attr["_executor_type"].i
        return super(TorchNpuGraph, self).load(ge_graph.SerializeToString(), options, input_placements, output_dtypes,
                                               executor_type)

    @pretty_error_msg
    def compile(self):
        """Compile the graph"""
        return super(TorchNpuGraph, self).compile()

    @pretty_error_msg
    def auto_tune(self, example_inputs=[], stream=None):
        """Compile the graph with aoe"""
        return super(TorchNpuGraph, self).auto_tune((example_inputs, stream))

    @pretty_error_msg
    def run(self, inputs, assigned_outputs=[], stream=None):
        """Run the graph"""
        return super(TorchNpuGraph, self).run((inputs, assigned_outputs, stream))
