import torch

from .common import _LoweringGuard, float_dtypes, byte_dtypes

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
_LoweringGuard.support(prims.convert_element_type, float_dtypes())
_LoweringGuard.support(aten.sigmoid, float_dtypes())
_LoweringGuard.support(aten.remainder, float_dtypes())
_LoweringGuard.support(aten.silu, float_dtypes())
_LoweringGuard.support(aten.neg, float_dtypes() + (torch.int32,))

# basic compare ops, support int32 as well
_LoweringGuard.support(aten.ge, float_dtypes() + (torch.int32,))
_LoweringGuard.support(aten.le, float_dtypes() + (torch.int32,))
_LoweringGuard.support(aten.gt, float_dtypes() + (torch.int32,))
_LoweringGuard.support(aten.lt, float_dtypes() + (torch.int32,))
_LoweringGuard.support(aten.eq, float_dtypes() + (torch.int32,))
_LoweringGuard.support(aten.ne, float_dtypes() + (torch.int32,))

# fill and create tensor ops
_LoweringGuard.support(aten.new_empty, float_dtypes())
_LoweringGuard.support(aten.detach, float_dtypes())
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
