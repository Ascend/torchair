from torchair._ge_concrete_graph.ge_converter.converter_utils import *


def split_float_int_tensor(self: List[Tensor]):
    int_idx = []
    int_tensor = []
    float_tensor = []
    for i, a in enumerate(self):
        if self[i].dtype in [DataType.DT_INT8, DataType.DT_UINT8, DataType.DT_INT16,
                             DataType.DT_INT32, DataType.DT_UINT32, DataType.DT_INT64, DataType.DT_UINT64]:
            int_idx.append(i)
            int_tensor.append(a)
        else:
            float_tensor.append(a)
    return int_idx, int_tensor, float_tensor


@declare_supported([
    Support([F32(2, 2)]),
    Support([F16(2, 2), I32(2, 2), BF16(2, 2)]),
])
@register_fx_node_ge_converter(torch.ops.aten._foreach_floor.default)
def conveter_aten__foreach_floor_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_floor(Tensor[] self) -> Tensor[]"""
    int_id, int_tensor, float_tensor = split_float_int_tensor(self)
    round_mode = ge.Fill(1, ge.Cast(2, dst_type=DataType.DT_INT8))
    if float_tensor:
        out = ge.ForeachRoundOffNumber(float_tensor, roundMode=round_mode)
        int_idx = range(len(int_id))
        for i in int_idx:
            out.insert(int_id[i], int_tensor[i])
        return out
    else:
        return self


@declare_supported([
    Support([F32(2, 2), F16(2, 2), BF16(2, 2)]),
    Support([I32(2, 2), I64(2, 2)]),
])
@register_fx_node_ge_converter(torch.ops.aten._foreach_frac.default)
def conveter_aten__foreach_frac_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_frac(Tensor[] self) -> Tensor[]"""
    int_id, int_tensor, float_tensor = split_float_int_tensor(self)
    round_mode = ge.Fill(1, ge.Cast(7, dst_type=DataType.DT_INT8))
    out = ge.ForeachRoundOffNumber(float_tensor, roundMode=round_mode)
    int_idx = range(len(int_id))
    for i in int_idx:
        out.insert(int_id[i], ge.ZerosLike(int_tensor[i]))
    return out


@declare_supported([
    Support([F32(2, 2)]),
    Support([F16(2, 2), BF16(2, 2)]),
])
@register_fx_node_ge_converter(torch.ops.aten._foreach_ceil.default)
def conveter_aten__foreach_ceil_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_ceil(Tensor[] self) -> Tensor[]"""
    round_mode = ge.Fill(1, ge.Cast(3, dst_type=DataType.DT_INT8))
    return ge.ForeachRoundOffNumber(self, roundMode=round_mode)


@declare_supported([
    Support([F32(2, 2), F16(2, 2), BF16(2, 2)]),
    Support([I32(2, 2), I64(2, 2)]),
])
@register_fx_node_ge_converter(torch.ops.aten._foreach_round.default)
def conveter_aten__foreach_round_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_round(Tensor[] self) -> Tensor[]"""
    int_id, int_tensor, float_tensor = split_float_int_tensor(self)
    round_mode = ge.Fill(1, ge.Cast(4, dst_type=DataType.DT_INT8))
    if float_tensor:
        out = ge.ForeachRoundOffNumber(float_tensor, roundMode=round_mode)
        int_idx = range(len(int_id))
        for i in int_idx:
            out.insert(int_id[i], int_tensor[i])
        return out
    else:
        return self


@declare_supported([
    Support([F32(2, 2)]),
    Support([F16(2, 2), BF16(2, 2)]),
])
@register_fx_node_ge_converter(torch.ops.aten._foreach_trunc.default)
def conveter_aten__foreach_trunc_default(self: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_trunc(Tensor[] self) -> Tensor[]"""
    round_mode = ge.Fill(1, ge.Cast(5, dst_type=DataType.DT_INT8))
    return ge.ForeachRoundOffNumber(self, roundMode=round_mode)
