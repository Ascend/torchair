from typing import List
import functools

import torch
from torch._inductor.virtualized import V
from inductor_npu_ext.common.asc_graph import _Tensor, _Op, ASCGraph


class _AscOpWrapper:
    def __init__(self, f):
        self._op = f.__name__
        self._f = f

    def _set_loop(self, tensors, loop=None):
        if self._op in ["data", "output", "workspace"]:
            return
        loop = loop if loop else V.kernel.contiguous_loop
        tensors: List[_Tensor] = tensors if isinstance(tensors, (list, tuple)) else [tensors]
        for tensor in tensors:
            tensor.as_loop(loop)

    def __call__(self, *args, loop=None, **kwargs):
        graph: ASCGraph = V.kernel.graph
        cache_key = f"{self._op}({args},{loop},{kwargs})"
        if cache_key in graph.op_cache:
            return graph.op_cache[cache_key]
        tensors = self._f(*args, **kwargs)
        if isinstance(tensors, (list, tuple)):
            tensors = (_Tensor(t) if not isinstance(t, _Tensor) else t for t in tensors)
            self._set_loop(tensors, loop)
            graph.op_cache[cache_key] = tensors
            return tensors
        tensor = _Tensor(tensors) if not isinstance(tensors, _Tensor) else tensors
        self._set_loop(tensor, loop)
        graph.op_cache[cache_key] = tensor
        return tensor


def asc_ops(fn):
    def wrapper(*args, **kwargs):
        return _AscOpWrapper(fn)(*args, **kwargs)
    return wrapper


def Op(t) -> _Op:
    return V.kernel.graph.add_op(t)


@asc_ops
def abs(x):
    op = Op("Abs")
    op.x = x
    return op.y


@asc_ops
def add(x1, x2):
    op = Op("Add")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def sub(x1, x2):
    op = Op("Sub")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def ge(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Ge")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def gt(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Gt")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def le(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Le")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def lt(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Lt")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def eq(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Eq")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def ne(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Ne")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def mul(x1, x2):
    op = Op("Mul")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def sigmoid(x):
    op = Op("Sigmoid")
    op.x = x
    return op.y


@asc_ops
def sqrt(x):
    op = Op("Sqrt")
    op.x = x
    return op.y


@asc_ops
def log(x):
    op = Op("Ln")
    op.x = x
    return op.y


@asc_ops
def rsqrt(x):
    op = Op("Rsqrt")
    op.x = x
    return op.y


@asc_ops
def neg(x):
    op = Op("Neg")
    op.x = x
    return op.y


@asc_ops
def maximum(x1, x2):
    op = Op("Maximum")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def exp(x):
    op = Op("Exp")
    op.x = x
    return op.y


@asc_ops
def broadcast(x):
    op = Op("Broadcast")
    op.x = x
    return op.y


@asc_ops
def transpose(x):
    op = Op("Transpose")
    op.x = x
    return op.y


@asc_ops
def reciprocal(x):
    op = Op("Reciprocal")
    op.x = x
    return op.y


@asc_ops
def truediv(x1, x2):
    if x1 == '1':
        return reciprocal(x2)
    op = Op("TrueDiv")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def div(x1, x2):
    return truediv(x1, x2)


@asc_ops
def square(x):
    return mul(x, x)


@asc_ops
def cast(x, *, dst, src=None, use_compute_types=True):
    op = Op("Cast")
    op.x = x
    op.y.dtype = dst
    return op.y


@asc_ops
def constant(value: str, dtype):
    op = Op("Scalar")
    op.attr.ir_attr.value = value
    op.y.dtype = dtype
    return op.y


@asc_ops
def reduction(x, *, reduce_type):
    op = Op(reduce_type.capitalize())
    op.x = x
    op.set_private_attr("compute_type", "reduce")
    return op.y


@asc_ops
def data(*, name, dtype, index):
    # drop name for stable cache hint
    op = Op("Data")
    op.attr.ir_attr.index = index
    op.y.dtype = dtype
    return op.y


@asc_ops
def output(*, name, src, dtype, index):
    # drop name for stable cache hint
    op = Op("Output")
    op.attr.ir_attr.index = index
    op.x = src
    op.y.dtype = dtype
    return op.y


@asc_ops
def workspace(*, name, src, dtype):
    op = Op("Workspace", name=name)
    op.x = src
    op.y.dtype = dtype
    return op.y


@asc_ops
def load(buffer, *, offset):
    op = Op("Load")
    if offset is not None:
        op.attr.ir_attr.offset = offset
    op.x = buffer
    return op.y


@asc_ops
def store(value):
    op = Op("Store")
    op.x = value
    return op.y


@asc_ops
def bitwise_and(x1, x2):
    op = Op("BitwiseAnd")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def pow(x1, x2):
    op = Op("Pow")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def where(x1, x2, x3):
    op = Op("Where")
    op.x1 = x1
    op.x2 = x2
    op.x3 = x3
    return op.y


# ---- 以下批量补齐 pyascir.h REGISTERED_OPS 中已暴露但 wrapper 缺失的算子。
#      命名对照见 ascir_builtin_ops.cpp 中的 REG_ASC_IR(<PascalCase>)。

# ---- unary elementwise (x → y, 输出 dtype 跟输入一致) ----

@asc_ops
def exp2(x):
    op = Op("Exp2")
    op.x = x
    return op.y


@asc_ops
def expm1(x):
    # PyTorch ops.expm1 → ascir Expm
    op = Op("Expm")
    op.x = x
    return op.y


@asc_ops
def floor(x):
    op = Op("Floor")
    op.x = x
    return op.y


@asc_ops
def ceil(x):
    op = Op("Ceil")
    op.x = x
    return op.y


@asc_ops
def round(x):
    op = Op("Round")
    op.x = x
    return op.y


@asc_ops
def trunc(x):
    op = Op("Trunc")
    op.x = x
    return op.y


@asc_ops
def sign(x):
    op = Op("Sign")
    op.x = x
    return op.y


@asc_ops
def erf(x):
    op = Op("Erf")
    op.x = x
    return op.y


@asc_ops
def erfc(x):
    op = Op("Erfc")
    op.x = x
    return op.y


@asc_ops
def erfcx(x):
    op = Op("Erfcx")
    op.x = x
    return op.y


@asc_ops
def digamma(x):
    op = Op("Digamma")
    op.x = x
    return op.y


@asc_ops
def lgamma(x):
    op = Op("Lgamma")
    op.x = x
    return op.y


@asc_ops
def log2(x):
    op = Op("Log2")
    op.x = x
    return op.y


@asc_ops
def log10(x):
    op = Op("Log10")
    op.x = x
    return op.y


@asc_ops
def log1p(x):
    op = Op("Log1p")
    op.x = x
    return op.y


@asc_ops
def tanh(x):
    op = Op("Tanh")
    op.x = x
    return op.y


@asc_ops
def sin(x):
    op = Op("Sin")
    op.x = x
    return op.y


@asc_ops
def cos(x):
    op = Op("Cos")
    op.x = x
    return op.y


@asc_ops
def sinh(x):
    op = Op("Sinh")
    op.x = x
    return op.y


@asc_ops
def cosh(x):
    op = Op("Cosh")
    op.x = x
    return op.y


@asc_ops
def tan(x):
    op = Op("Tan")
    op.x = x
    return op.y


@asc_ops
def asin(x):
    op = Op("Asin")
    op.x = x
    return op.y


@asc_ops
def asinh(x):
    op = Op("Asinh")
    op.x = x
    return op.y


@asc_ops
def acos(x):
    op = Op("Acos")
    op.x = x
    return op.y


@asc_ops
def acosh(x):
    op = Op("Acosh")
    op.x = x
    return op.y


@asc_ops
def atan(x):
    op = Op("Atan")
    op.x = x
    return op.y


@asc_ops
def atanh(x):
    op = Op("Atanh")
    op.x = x
    return op.y


@asc_ops
def isnan(x):
    # 输出 uint8（asc 用 uint8 表 bool）
    op = Op("Isnan")
    op.x = x
    op.y.dtype = torch.uint8
    return op.y


@asc_ops
def isfinite(x):
    op = Op("IsFinite")
    op.x = x
    op.y.dtype = torch.uint8
    return op.y


@asc_ops
def relu(x):
    op = Op("Relu")
    op.x = x
    return op.y


@asc_ops
def gelu(x):
    op = Op("Gelu")
    op.x = x
    return op.y


@asc_ops
def leaky_relu(x, negative_slope=0.01):
    op = Op("LeakyRelu")
    op.x = x
    op.attr.ir_attr.negative_slope = negative_slope
    return op.y


@asc_ops
def bitwise_not(x):
    op = Op("BitwiseNot")
    op.x = x
    return op.y


@asc_ops
def logical_not(x):
    op = Op("LogicalNot")
    op.x = x
    op.y.dtype = torch.uint8
    return op.y


@asc_ops
def nop(x):
    op = Op("Nop")
    op.x = x
    return op.y


# ---- unary type-conversion (x:T1 → y:T2，输出 dtype 显式指定) ----

@asc_ops
def floor_to_int(x, *, dst=torch.int32):
    op = Op("FloorToInt")
    op.x = x
    op.y.dtype = dst
    return op.y


@asc_ops
def ceil_to_int(x, *, dst=torch.int32):
    # ascir 拼写: Ceil2Int
    op = Op("Ceil2Int")
    op.x = x
    op.y.dtype = dst
    return op.y


@asc_ops
def round_to_int(x, *, dst=torch.int32):
    op = Op("RoundToInt")
    op.x = x
    op.y.dtype = dst
    return op.y


@asc_ops
def trunc_to_int(x, *, dst=torch.int32):
    op = Op("TruncToInt")
    op.x = x
    op.y.dtype = dst
    return op.y


# ---- binary elementwise (x1, x2 → y) ----

@asc_ops
def minimum(x1, x2):
    op = Op("Minimum")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def atan2(x1, x2):
    op = Op("Atan2")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def copysign(x1, x2):
    op = Op("CopySign")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def hypot(x1, x2):
    op = Op("Hypot")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def fmod(x1, x2):
    op = Op("Fmod")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def mod(x1, x2):
    op = Op("Mod")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def remainder(x1, x2):
    op = Op("Remainder")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def floordiv(x1, x2):
    op = Op("FloorDiv")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def truncdiv(x1, x2):
    op = Op("TruncDiv")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def axpy(x1, x2):
    # ascir Axpy: y = alpha*x1 + x2，通过 attr 控制 alpha；这里只暴露最简形
    op = Op("Axpy")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def bitwise_or(x1, x2):
    op = Op("BitwiseOr")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def bitwise_xor(x1, x2):
    op = Op("BitwiseXor")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def xor(x1, x2):
    op = Op("Xor")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def logical_and(x1, x2):
    op = Op("LogicalAnd")
    op.x1 = x1
    op.x2 = x2
    op.y.dtype = torch.uint8
    return op.y


@asc_ops
def logical_or(x1, x2):
    op = Op("LogicalOr")
    op.x1 = x1
    op.x2 = x2
    op.y.dtype = torch.uint8
    return op.y


@asc_ops
def logical_xor(x1, x2):
    op = Op("LogicalXor")
    op.x1 = x1
    op.x2 = x2
    op.y.dtype = torch.uint8
    return op.y


@asc_ops
def bitwise_left_shift(x1, x2):
    op = Op("LShift")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def bitwise_right_shift(x1, x2):
    op = Op("RShift")
    op.x1 = x1
    op.x2 = x2
    return op.y


# inductor NPUOverrides 用的别名
lshift = bitwise_left_shift
rshift = bitwise_right_shift


# ---- ternary (x1, x2, x3 → y) ----

@asc_ops
def fma(x1, x2, x3):
    # y = x1*x2 + x3
    op = Op("Fma")
    op.x1 = x1
    op.x2 = x2
    op.x3 = x3
    return op.y


@asc_ops
def select(cond, x_true, x_false):
    # ascir Select: T1=uint8 (cond), T2=fp/int (value)，跟 Where 语义相同；
    # 保留 select 是因为 ascir 把它跟 Where 注册成两个 op (cf. pyascir.h)
    op = Op("Select")
    op.x1 = cond
    op.x2 = x_true
    op.x3 = x_false
    return op.y


@asc_ops
def clip_by_value(x, clip_min, clip_max):
    # ascir ClipByValue：把 x 钳制到 [clip_min, clip_max]
    op = Op("ClipByValue")
    op.x1 = x
    op.x2 = clip_min
    op.x3 = clip_max
    return op.y


@asc_ops
def _unsupported_op(*args, _op=None, **kwargs):
    op = V.kernel.graph.add_op(_op, is_unsupported=True)
    for i, arg in enumerate(args):
        if isinstance(arg, _Tensor):
            setattr(op, f"x{i+1}", arg)
        else:
            op.ir_attr.__setattr__(f"input{i+1}", str(arg))
    for k, v in kwargs.items():
        op.ir_attr.__setattr__(k, str(v))
    return op.y


def _unsupported(*args, _op=None, **kwargs):
    from inductor_npu_ext.config import _debug_options

    if "nothrow" in _debug_options:
        return _unsupported_op(*args, _op=_op, **kwargs)

    raise NotImplementedError(f"Asc op '{_op}' is not implemented yet, args: {args}, kwargs: {kwargs}")


def __getattr__(name):
    return functools.partial(_unsupported, _op=name)
