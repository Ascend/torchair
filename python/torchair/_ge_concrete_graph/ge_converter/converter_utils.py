from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)
import sys
import operator
import math
from packaging import version

import numpy as np
import torch
import torch._prims_common as utils
from torch._ops import OpOverload
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge import attr, Clone
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.compat_ir import ge_op
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported, \
    register_checkpoint_func, torch_type_to_ge_type
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec, get_ge_rng_state, is_sym, \
    _ge_dtype_to_ge_proto_dtype as ge_dtype_to_ge_proto_dtype, ge_type_to_torch_type, torch_type_to_ge_type, \
    assert_args_checkout
from torchair._ge_concrete_graph.utils import dtype_promote, specific_op_input_layout, specific_op_output_layout, \
    force_op_unknown_shape, normalize_max_value, normalize_min_value, _display_ge_type as display_ge_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, \
    I8, U8, BOOL, T, C64, Support
from torchair._ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType
from torchair._ge_concrete_graph.utils import get_cann_opp_version
from torchair._utils.check_platform import is_arch35
from torchair.core.utils import logger
from torchair.ge.ge_custom import custom_op