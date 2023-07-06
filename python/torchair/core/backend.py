#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions used for NPU device"""

import os
import threading
import logging
import atexit
import sys
from typing import Dict

from . import _torchair


def stupid_repeat(word, times):
    return _torchair.StupidRepeat(word, times)


def _try_get_torch_npu_device():
    if 'torch_npu' not in sys.modules:
        return None

    from . import _npu_graph_executor

    torch_npu_module = sys.modules['torch_npu']
    if torch_npu_module.npu.is_initialized():
        return torch_npu_module.npu.current_device()
    else:
        print(f"[WARNNING] torch_npu imported but not initialized.")
        return None


def initialize_graph_engine():
    options: Dict[str, str] = {}
    torch_npu_device = _try_get_torch_npu_device()
    if torch_npu_device is not None:
        options['ge.exec.deviceId'] = str(torch_npu_device)
    else:
        options['ge.exec.deviceId'] = os.getenv('ASCEND_DEVICE_ID', '0')
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
