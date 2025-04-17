from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._fused_moving_avg_obs_fq_helper.default)
def conveter_aten__fused_moving_avg_obs_fq_helper_default(
    self: Tensor,
    observer_on: Tensor,
    fake_quant_on: Tensor,
    running_min: Tensor,
    running_max: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    averaging_const: float,
    quant_min: int,
    quant_max: int,
    ch_axis: int,
    per_row_fake_quant: bool = False,
    symmetric_quant: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)"""
    raise NotImplementedError("torch.ops.aten._fused_moving_avg_obs_fq_helper.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._fused_moving_avg_obs_fq_helper.out)
def conveter_aten__fused_moving_avg_obs_fq_helper_out(
    self: Tensor,
    observer_on: Tensor,
    fake_quant_on: Tensor,
    running_min: Tensor,
    running_max: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    averaging_const: float,
    quant_min: int,
    quant_max: int,
    ch_axis: int,
    per_row_fake_quant: bool = False,
    symmetric_quant: bool = False,
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_fused_moving_avg_obs_fq_helper.out(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False, *, Tensor(e!) out0, Tensor(f!) out1) -> (Tensor(e!), Tensor(f!))"""
    raise NotImplementedError("torch.ops.aten._fused_moving_avg_obs_fq_helper.out ge_converter is not implemented!")
