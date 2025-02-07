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

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, register_checkpoint_func
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, get_ge_rng_state
from torchair._ge_concrete_graph.utils import dtype_promote


@register_checkpoint_func(torch.ops.aten.native_dropout.default)
def native_dropout_checkpoint(
    tensor_input: Tensor,
    p: float,
    train: Optional[bool],
    meta_outputs: TensorSpec = None,
    rng_state: Optional[Tensor] = None
):
    if train is None or train == True:
        prob = 1. - p
        prob = dtype_promote(prob, target_dtype=tensor_input.dtype)
        shape = ge.Shape(tensor_input)
        if rng_state is None:
            seed, offset = get_ge_rng_state(philox_num=10)
        else:
            seed, offset = ge.Unpack(rng_state, num=2, axis=0)
        # DropOutGenMask use seed and seed1 to generator a seed, list this:
        # seed1    seed
        # 127-64   63-0
        # so, we set seed1 = 0 to ensure the seed which user set is equal to the seed
        # used by the oprerator DropOutGenMask.
        seed1 = ge.Const(0, dtype=DataType.DT_INT64)
        offset0 = ge.Const(0, dtype=DataType.DT_INT64)
        # offset is similar to seed.
        offset_list = ge.ConcatV2([offset0, offset], 0, N=2)
        mask = ge.StatelessDropOutGenMask(shape, prob, seed, seed1, offset_list)
        return (seed, offset), (ge.DropOutDoMask(tensor_input, mask, prob), mask)
    else:
        mask = ge.Fill(ge.Shape(tensor_input), ge.Cast(1., dst_type=DataType.DT_BOOL))
        return (None, None), (tensor_input, mask)


# No testcase because the dtype and shape of output *mask* are different from cpu's.
@register_fx_node_ge_converter(torch.ops.aten.native_dropout.default)
def conveter_aten_native_dropout_default(
    input: Tensor, p: float, train: Optional[bool], meta_outputs: TensorSpec = None
):
    """NB: aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)"""
    _, result = native_dropout_checkpoint(input, p, train, meta_outputs, None)
    return result


@register_fx_node_ge_converter(torch.ops.aten.native_dropout.out)
def conveter_aten_native_dropout_out(
    input: Tensor,
    p: float,
    train: Optional[bool],
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::native_dropout.out(Tensor input, float p, bool? train, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))"""
    raise NotImplementedError("torch.ops.aten.native_dropout.out ge_converter is not implemented!")
