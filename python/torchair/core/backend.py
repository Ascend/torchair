"""Functions used for NPU device"""

import os
import threading
import logging
import atexit
import sys
from typing import Dict

import torch
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
        op_compile_config['ge.exec.allow_hf32'] = '10' # enable conv hf32 as default
        op_compile_config['ge.deterministic'] = '1'
        return op_compile_config

    torch_npu_module = sys.modules['torch_npu']
    mm_hf32 = '1' if torch_npu_module.npu.matmul.allow_hf32 else '0'
    conv_hf32 = '1' if torch_npu_module.npu.conv.allow_hf32 else '0'
    op_compile_config['ge.exec.allow_hf32'] = conv_hf32 + mm_hf32
    op_compile_config['ge.deterministic'] = '1' if torch.are_deterministic_algorithms_enabled() else '0'
    return op_compile_config


def initialize_graph_engine():
    options: Dict[str, str] = {}
    torch_npu_device = _try_get_torch_npu_device()
    if torch_npu_device is not None:
        options['ge.exec.deviceId'] = str(torch_npu_device)
        options['ge_run_with_torch_npu'] = "1"
    else:
        options['ge.exec.deviceId'] = os.getenv('ASCEND_DEVICE_ID', '0')
        options['ge_run_with_torch_npu'] = "0"

    options.update(_get_global_op_compile_config())
    _torchair.InitializeGraphEngine(options)


def finalize_graph_engine():
    _torchair.FinalizeGraphEngine()


atexit.register(finalize_graph_engine)


class TorchNpuGraph(_torchair.TorchNpuGraphBase):
    def __init__(self, name=""):
        super(TorchNpuGraph, self).__init__(str(name))

    def load(self, serialized_ge_graph, options={}):
        """Load the graph"""
        return super(TorchNpuGraph, self).load(serialized_ge_graph, options)

    def compile(self):
        """Compile the graph"""
        return super(TorchNpuGraph, self).compile()

    def auto_tune(self, example_inputs=[], stream=None):
        """Compile the graph with aoe"""
        return super(TorchNpuGraph, self).auto_tune((example_inputs, stream))

    def run(self, inputs, assigned_outputs=[], stream=None):
        """Run the graph"""
        return super(TorchNpuGraph, self).run((inputs, assigned_outputs, stream))
