import torch

from .common import _LoweringGuard, float_dtypes, byte_dtypes, Soc

aten = torch.ops.aten
prims = torch.ops.prims

# basic math ops
_LoweringGuard.support(aten.add, float_dtypes())
_LoweringGuard.support(aten.exp, float_dtypes())
_LoweringGuard.support(aten.mul, float_dtypes())
_LoweringGuard.support(aten.pow, float_dtypes())
_LoweringGuard.support(aten.div, float_dtypes())
_LoweringGuard.support(aten.rsqrt, float_dtypes())
_LoweringGuard.support(aten.sqrt, float_dtypes())
_LoweringGuard.support(aten.sub, float_dtypes())
_LoweringGuard.support(aten.abs, float_dtypes())
_LoweringGuard.support(aten.floor_divide, float_dtypes())
# convert_element_type 需接受 bool 输入：比较 op 输出 bool，下游常接
# convert(bool→float)；否则 in dtype=bool 会被 dtype 检查拦回。
_LoweringGuard.support(prims.convert_element_type, float_dtypes() + byte_dtypes() + (torch.int32,))
_LoweringGuard.support(aten.sigmoid, float_dtypes())
_LoweringGuard.support(aten.remainder, float_dtypes())
_LoweringGuard.support(aten.silu, float_dtypes())
_LoweringGuard.support(aten.log1p, float_dtypes(), since=Soc.A5)
_LoweringGuard.support(aten.relu, float_dtypes())
# aten.sgn 在实数路径下被 inductor decomposition 改写成 aten.sign，所以两个都要放行
_LoweringGuard.support(aten.sgn, float_dtypes())
_LoweringGuard.support(aten.sign, float_dtypes())
_LoweringGuard.support(aten.neg, float_dtypes() + (torch.int32,))

# basic compare ops, support int32 as well。
# 比较 op 输入 float/int32，输出 bool —— 必须显式给 support_out_dtypes 放行 bool/uint8，
# 否则输出 dtype 检查会把它们拦成 fallback。
_cmp_in = float_dtypes() + (torch.int32,)
_cmp_out = byte_dtypes()  # (uint8, bool)
_LoweringGuard.support(aten.ge, _cmp_in, support_out_dtypes=_cmp_out)
_LoweringGuard.support(aten.le, _cmp_in, support_out_dtypes=_cmp_out)
_LoweringGuard.support(aten.gt, _cmp_in, support_out_dtypes=_cmp_out)
_LoweringGuard.support(aten.lt, _cmp_in, support_out_dtypes=_cmp_out)
_LoweringGuard.support(aten.eq, _cmp_in, support_out_dtypes=_cmp_out)
_LoweringGuard.support(aten.ne, _cmp_in, support_out_dtypes=_cmp_out)

# fill and create tensor ops
_LoweringGuard.support(aten.new_empty, float_dtypes())
_LoweringGuard.support(aten.arange, float_dtypes())
_LoweringGuard.support(aten.copy_, float_dtypes())
_LoweringGuard.support(aten.copy, float_dtypes())
_LoweringGuard.support(aten.zeros_like, float_dtypes())
_LoweringGuard.support(aten.zeros, float_dtypes())
_LoweringGuard.support(aten._to_copy, float_dtypes())

# bitwise ops
_LoweringGuard.support(aten.bitwise_and, byte_dtypes())

# reduction ops
_LoweringGuard.support(aten.sum, float_dtypes())
_LoweringGuard.support(aten.mean, float_dtypes())
_LoweringGuard.support(aten.max, float_dtypes())
_LoweringGuard.support(aten.min, float_dtypes())

# view ops
_LoweringGuard.support(aten.unsqueeze, float_dtypes())
_LoweringGuard.support(aten.squeeze, float_dtypes())
_LoweringGuard.support(aten.permute, float_dtypes())
_LoweringGuard.support(aten.select, float_dtypes())
_LoweringGuard.support(aten.slice, float_dtypes())
_LoweringGuard.support(aten._unsafe_view, float_dtypes())
_LoweringGuard.support(aten.t, float_dtypes())
_LoweringGuard.support(aten.transpose, float_dtypes())
_LoweringGuard.support(aten.expand, float_dtypes())
_LoweringGuard.support(aten.alias, float_dtypes())
_LoweringGuard.support(aten.sym_size, float_dtypes())
_LoweringGuard.support(aten.detach, float_dtypes())
