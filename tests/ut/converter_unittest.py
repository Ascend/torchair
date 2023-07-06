#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torchair as tng
from torchair.ge_concrete_graph.fx2ge_converter import test_converter, Converter
from torchair import CompilerConfig
aten = torch.ops.aten

config = CompilerConfig()
config.aoe_config.aoe_mode = "1"
config.debug.graph_dump.type = "pbtxt"

npu_backend = tng.get_npu_backend(compiler_config=config)


def check_tensor_same(a, b):
    assert a.dtype == b.dtype, f"Datatype mismatch {a.dtype} vs. {b.dtype}"
    assert a.size() == b.size(), f"Shape mismatch {a.size()} vs. {b.size()}"

    if a.dtype in (torch.float16, torch.float32, torch.float64):
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-5), f"Value mismatch {a} vs. {b}"
    else:
        assert torch.all(a == b), f"Value mismatch {a} vs. {b}"

def check_result(compiled_rets, eager_rets):
    assert type(compiled_rets) == type(eager_rets), f"result type mismatch {type(compiled_rets)} vs. {type(eager_rets)}"
    compiled_rets = (compiled_rets, ) if isinstance(compiled_rets, torch.Tensor) else compiled_rets
    eager_rets = (eager_rets, ) if isinstance(eager_rets, torch.Tensor) else eager_rets
    for c_ret, e_ret in zip(compiled_rets, eager_rets):
        assert type(c_ret) == type(e_ret), f"result type mismatch {type(c_ret)} vs. {type(e_ret)}"
        if isinstance(c_ret, (list, tuple)):
            for c_tensor, e_tensor in zip(c_ret, e_ret):
                check_tensor_same(c_tensor, e_tensor)
        else:
            assert isinstance(c_ret, torch.Tensor)
            check_tensor_same(c_ret, e_ret)


Converter.compile_backend = npu_backend
Converter.result_checker = check_result

test_converter()
