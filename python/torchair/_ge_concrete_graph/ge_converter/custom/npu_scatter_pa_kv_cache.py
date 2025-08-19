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
from torchair._ge_concrete_graph.supported_declaration \
import _TypedTensor, F32, F16, I32, I16, I64, I8, U8, BF16, Support


@declare_supported(
    [
        Support(I8(16, 4, 32), F16(16, 4, 32), I32(16), compress_lens=I32(1), compress_seq_offsets=I32(1),
                seq_lens=I32(1), cache_mode="PA_NZ", out=(I8(2, 8, 32, 16), F16(2, 16, 32, 16))),
        Support(I8(16, 4, 32), BF16(16, 4, 32), I32(16), compress_lens=I32(1), compress_seq_offsets=I32(1),
                seq_lens=I32(1), cache_mode="PA_NZ", out=(I8(2, 8, 32, 16), BF16(2, 16, 32, 16))),
        Support(F16(16, 4, 32), F16(16, 4, 32), I32(16), compress_lens=I32(1), compress_seq_offsets=I32(1),
                seq_lens=I32(1), cache_mode="PA_NZ", out=(F16(2, 8, 32, 16), F16(2, 16, 32, 16))),
        Support(F16(16, 4, 32), BF16(16, 4, 32), I32(16), compress_lens=I32(1), compress_seq_offsets=I32(1),
                seq_lens=I32(1), cache_mode="PA_NZ", out=(F16(2, 8, 32, 16), BF16(2, 16, 32, 16))),
        Support(BF16(16, 4, 32), F16(16, 4, 32), I32(2), compress_lens=I32(1), compress_seq_offsets=I32(1),
                seq_lens=I32(1), cache_mode="PA_NZ", out=(BF16(2, 8, 32, 16), F16(2, 16, 32, 16))),
        Support(BF16(16, 4, 32), BF16(16, 4, 32), I32(16), compress_lens=I32(1), compress_seq_offsets=I32(1),
                seq_lens=I32(1), cache_mode="PA_NZ", out=(BF16(2, 8, 32, 16), BF16(2, 16, 32, 16))),
        Support(I8(16, 4, 32), F16(16, 4, 32), I64(16), compress_lens=I64(1), compress_seq_offsets=I64(1),
                seq_lens=I64(1), cache_mode="PA_NZ", out=(I8(2, 8, 32, 16), F16(2, 16, 32, 16))),
        Support(I8(16, 4, 32), BF16(16, 4, 32), I64(16), compress_lens=I64(1), compress_seq_offsets=I64(1),
                seq_lens=I64(1), cache_mode="PA_NZ", out=(I8(2, 8, 32, 16), BF16(2, 16, 32, 16))),
        Support(F16(16, 4, 32), F16(16, 4, 32), I64(16), compress_lens=I64(1), compress_seq_offsets=I64(1),
                seq_lens=I64(1), cache_mode="PA_NZ", out=(F16(2, 8, 32, 16), F16(2, 16, 32, 16))),
        Support(F16(16, 4, 32), BF16(16, 4, 32), I64(16), compress_lens=I64(1), compress_seq_offsets=I64(1),
                seq_lens=I64(1), cache_mode="PA_NZ", out=(F16(2, 8, 32, 16), BF16(2, 16, 32, 16))),
        Support(BF16(16, 4, 32), F16(16, 4, 32), I64(16), compress_lens=I64(1), compress_seq_offsets=I64(1),
                seq_lens=I64(1), cache_mode="PA_NZ", out=(BF16(2, 8, 32, 16), F16(2, 16, 32, 16))),
        Support(BF16(16, 4, 32), BF16(16, 4, 32), I64(16), compress_lens=I64(1), compress_seq_offsets=I64(1),
                seq_lens=I64(1), cache_mode="PA_NZ", out=(BF16(2, 8, 32, 16), BF16(2, 16, 32, 16))),
    ]
)

@register_fx_node_ge_converter(torch.ops.npu.npu_scatter_pa_kv_cache_functional.default)
def conveter_npu_scatter_pa_kv_cache_functional(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    *,
    compress_lens: Optional[Tensor] = None,
    compress_seq_offsets: Optional[Tensor] = None,
    seq_lens: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):

    """
    NB: func: npu_scatter_pa_kv_cache_functional(Tensor key, Tensor value, Tensor key_cache, Tensor value_cache,
                                       Tensor slot_mapping, *, Tensor? compress_lens=None,
                                       Tensor? compress_seq_offsets=None, Tensor? seq_lens=None,
                                       str cache_mode='Norm') -> (Tensor, Tensor)
    """
    key_cache_copy = ge.TensorMove(key_cache)
    value_cache_copy = ge.TensorMove(value_cache)
    cache_mode = "PA_NZ"
    return ge.ScatterPaKvCache(key, key_cache_copy, slot_mapping, value, value_cache_copy,
                               compress_lens=compress_lens, compress_seq_offset=compress_seq_offsets,
                               seq_lens=seq_lens, cache_mode=cache_mode)
