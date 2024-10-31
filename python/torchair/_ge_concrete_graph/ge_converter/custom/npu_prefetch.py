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
from torchair.ge import attr
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.compat_ir import ge_op
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, get_default_ge_graph
from torchair._ge_concrete_graph.utils import dtype_promote, get_cann_opp_version
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support


@declare_supported([
    Support(F16(4, 8192), F16(4, 8192), 131072, 512),
    Support(F16(4, 8192), F16(4, 8192), 160000),
    Support(F32(4, 8192), F32(4, 8192), 1),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_prefetch.default)
def conveter_npu_prefetch_default(
    self: Tensor,
    dependency: Optional[Tensor],
    max_size: int,
    offset: int = 0,
    meta_outputs: TensorSpec = None,
):
    """NB: func: npu_prefetch(Tensor self, Tensor? dependency, int max_size) -> ()"""
    if max_size <= 0:
        raise ValueError(f"max_size should be greater than zero, but got {max_size}")
    
    if offset < 0:
        raise ValueError(f"offset should be nonnegative, but got {max_size}")
    
    opp_version = get_cann_opp_version()
    if (offset != 0) and opp_version.startswith('7.5'):
        raise NotImplementedError("torch.ops.npu.npu_prefetch.default unsupport offset attr.")

    if dependency is None:
        raise NotImplementedError("torch.ops.npu.npu_prefetch.default ge converter is not implement "
                                  "when dependency is None.")
    
    if not opp_version.startswith('7.5'):
        ge.Cmo(self, max_size=max_size, offset=offset, dependencies=[dependency])
    else:
        # Cmo does not have offset attr in 7.5 version, and can not use ge_op for compatibility check.
        inputs = {
            "src": self,
        }

        attrs = {
            "max_size": attr.Int(max_size),
            "type": attr.Int(6),
        }

        outputs = [
        ]
        ge_op(op_type="Cmo", inputs=inputs, outputs=outputs, attrs=attrs, dependencies=[dependency])

