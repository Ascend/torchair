#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
# ============================================================================

from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import sys
import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.npu.npu_scatter_pa_kv_cache_functional.default)
def conveter_npu_scatter_pa_kv_cache_functional(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    *,
    compress_lens: Optional[Tensor] = None,
    compress_seq_offset: Optional[Tensor] = None,
    seq_lens: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):

    """
    NB: func: npu_scatter_pa_kv_cache_functional(Tensor key, Tensor value, Tensor key_cache, Tensor value_cache,
                                       Tensor slot_mapping, *, Tensor? compress_lens=None,
                                       Tensor? compress_seq_offsets=None, Tensor? seq_lens=None,
                                       str cache_mode='Norm', str scatter_mode='None',
                                       int[] strides, int[] offsets) -> (Tensor, Tensor)
    """
    key_cache_copy = ge.TensorMove(key_cache)
    value_cache_copy = ge.TensorMove(value_cache)
    cache_mode = "PA_NZ"
    scatter_mode = "None"
    strides = [1, 1]
    offsets = [0, 0]
    return ge.ScatterPaKvCache(key, key_cache_copy, slot_mapping, value, value_cache_copy,
                               compress_lens=compress_lens, compress_seq_offset=compress_seq_offset,
                               seq_lens=seq_lens, cache_mode=cache_mode, scatter_mode=scatter_mode,
                               strides=strides, offsets=offsets)
