from typing import Callable, Optional, Union, Any

import sympy
import torch

from torch._inductor.codegen.common import OpOverrides
from torch._inductor.ir import ReductionType, StoreMode
from inductor_npu_ext.common.asc_graph import _Tensor as T
from inductor_npu_ext import asc_ops as ir


class NPUOverrides(OpOverrides):
    def __init__(self):
        super().__init__()

    # 算术运算类操作20个
    # ----------------------------------------------------------------------------------------------------
    # 计算绝对值, torch.ops.aten.abs
    @staticmethod
    def abs(x0: T) -> T:
        return ir.abs(x0)

    # 加法运算, torch.ops.aten.add
    @staticmethod
    def add(x0: T, x1: T) -> T:
        return ir.add(x0, x1)

    # 复制符号位, torch.ops.aten.copysign
    @staticmethod
    def copysign(x0: T, x1: T) -> T:
        return ir.copysign(x0, x1)

    # 地板除法, torch.ops.aten.floordiv
    @staticmethod
    def floordiv(x0: T, x1: T) -> T:
        return ir.floordiv(x0, x1)

    # 浮点取模, torch.ops.aten.fmod
    @staticmethod
    def fmod(x0: T, x1: T) -> T:
        return ir.fmod(x0, x1)

    # 整数真除法,
    @staticmethod
    def int_truediv(x0: T, x1: T) -> T:
        return ir.int_truediv(x0, x1)

    # 取模运算,
    @staticmethod
    def mod(x0: T, x1: T) -> T:
        return ir.mod(x0, x1)

    # 乘法运算, torch.ops.aten.mul
    @staticmethod
    def mul(x0: T, x1: T) -> T:
        return ir.mul(x0, x1)

    # 取负操作, torch.ops.aten.neg
    @staticmethod
    def neg(x0: T) -> T:
        return ir.neg(x0)

    # 下一个浮点数, torch.ops.aten.nextafter
    @staticmethod
    def nextafter(x0: T, x1: T) -> T:
        return ir.nextafter(x0, x1)

    # 幂运算, torch.ops.aten.pow
    @staticmethod
    def pow(x0: T, x1: T) -> T:
        return ir.pow(x0, x1)

    # 倒数, torch.ops.aten.reciprocal
    @staticmethod
    def reciprocal(x0: T) -> T:
        return ir.reciprocal(x0)

    # 余数运算, torch.ops.aten.remainder
    @staticmethod
    def remainder(x0: T, x1: T) -> T:
        return ir.remainder(x0, x1)

    # 平方根倒数, torch.ops.aten.rsqrt
    @staticmethod
    def rsqrt(x0: T) -> T:
        return ir.rsqrt(x0)

    # 符号函数, torch.ops.aten.sign
    @staticmethod
    def sign(x0: T) -> T:
        return ir.sign(x0)

    # 符号位判断, torch.ops.aten.signbit
    @staticmethod
    def signbit(x0: T) -> T:
        return ir.signbit(x0)

    # 平方运算, torch.ops.aten.square
    @staticmethod
    def square(x0: T) -> T:
        return ir.square(x0)

    # 减法运算, torch.ops.aten.sub
    @staticmethod
    def sub(x0: T, x1: T) -> T:
        return ir.sub(x0, x1)

    # 真除法,
    @staticmethod
    def truediv(x0: T, x1: T) -> T:
        return ir.truediv(x0, x1)

    # 截断除法,
    @staticmethod
    def truncdiv(x0: T, x1: T) -> T:
        return ir.truncdiv(x0, x1)

    # 三角函数类操作7个
    # ----------------------------------------------------------------------------------------------------
    # 反余弦函数, torch.ops.aten.acos

    @staticmethod
    def acos(x0: T) -> T:
        return ir.acos(x0)

    # 反正弦函数, torch.ops.aten.asin
    @staticmethod
    def asin(x0: T) -> T:
        return ir.asin(x0)

    # 反正切函数, torch.ops.aten.atan
    @staticmethod
    def atan(x0: T) -> T:
        return ir.atan(x0)

    # 四象限反正切, torch.ops.aten.atan2
    @staticmethod
    def atan2(x0: T, x1: T) -> T:
        return ir.atan2(x0, x1)

    # 余弦函数, torch.ops.aten.cos
    @staticmethod
    def cos(x0: T) -> T:
        return ir.cos(x0)

    # 正弦函数, torch.ops.aten.sin
    @staticmethod
    def sin(x0: T) -> T:
        return ir.sin(x0)

    # 正切函数, torch.ops.aten.tan
    @staticmethod
    def tan(x0: T) -> T:
        return ir.tan(x0)

    # 双曲函数类操作6个
    # ----------------------------------------------------------------------------------------------------
    # 反双曲余弦, torch.ops.aten.acosh

    @staticmethod
    def acosh(x0: T) -> T:
        return ir.acosh(x0)

    # 反双曲正弦, torch.ops.aten.asinh
    @staticmethod
    def asinh(x0: T) -> T:
        return ir.asinh(x0)

    # 反双曲正切, torch.ops.aten.atanh
    @staticmethod
    def atanh(x0: T) -> T:
        return ir.atanh(x0)

    # 双曲余弦, torch.ops.aten.cosh
    @staticmethod
    def cosh(x0: T) -> T:
        return ir.cosh(x0)

    # 双曲正弦, torch.ops.aten.sinh
    @staticmethod
    def sinh(x0: T) -> T:
        return ir.sinh(x0)

    # 双曲正切, torch.ops.aten.tanh
    @staticmethod
    def tanh(x0: T) -> T:
        return ir.tanh(x0)

    # 特殊函数(torch.special)类操作40个
    # ----------------------------------------------------------------------------------------------------
    # 艾里函数Ai, torch.special.airy_ai

    @staticmethod
    def airy_ai(x: T) -> T:
        return ir.airy_ai(x)

    # 第一类贝塞尔函数0阶, torch.special.bessel_j0
    @staticmethod
    def bessel_j0(x: T) -> T:
        return ir.bessel_j0(x)

    # 第一类贝塞尔函数1阶, torch.special.bessel_j1
    @staticmethod
    def bessel_j1(x: T) -> T:
        return ir.bessel_j1(x)

    # 第二类贝塞尔函数0阶, torch.special.bessel_y0
    @staticmethod
    def bessel_y0(x: T) -> T:
        return ir.bessel_y0(x)

    # 第二类贝塞尔函数1阶, torch.special.bessel_y1
    @staticmethod
    def bessel_y1(x: T) -> T:
        return ir.bessel_y1(x)

    # 切比雪夫多项式T, torch.special.chebyshev_polynomial_t
    @staticmethod
    def chebyshev_polynomial_t(x: T, y: T) -> T:
        return ir.chebyshev_polynomial_t(x, y)

    # 切比雪夫多项式U, torch.special.chebyshev_polynomial_u
    @staticmethod
    def chebyshev_polynomial_u(x: T, y: T) -> T:
        return ir.chebyshev_polynomial_u(x, y)

    # 切比雪夫多项式V, torch.special.chebyshev_polynomial_v
    @staticmethod
    def chebyshev_polynomial_v(x: T, y: T) -> T:
        return ir.chebyshev_polynomial_v(x, y)

    # 切比雪夫多项式W, torch.special.chebyshev_polynomial_w
    @staticmethod
    def chebyshev_polynomial_w(x: T, y: T) -> T:
        return ir.chebyshev_polynomial_w(x, y)

    # 双伽马函数, torch.ops.aten.digamma
    @staticmethod
    def digamma(x: T) -> T:
        return ir.digamma(x)

    # 缩放互补误差函数, torch.special.erfcx
    @staticmethod
    def erfcx(x: T) -> T:
        return ir.erfcx(x)

    # 融合乘加,
    @staticmethod
    def fma(x: T, y: T, z: T) -> T:
        return ir.fma(x, y, z)

    # 下不完全伽马函数, torch.special.gammainc
    @staticmethod
    def gammainc(x: T, y: T) -> T:
        return ir.gammainc(x, y)

    # 上不完全伽马函数, torch.special.gammaincc
    @staticmethod
    def gammaincc(x: T, y: T) -> T:
        return ir.gammaincc(x, y)

    # 埃尔米特多项式H, torch.special.hermite_polynomial_h
    @staticmethod
    def hermite_polynomial_h(x: T, y: T) -> T:
        return ir.hermite_polynomial_h(x, y)

    # 埃尔米特多项式He, torch.special.hermite_polynomial_he
    @staticmethod
    def hermite_polynomial_he(x: T, y: T) -> T:
        return ir.hermite_polynomial_he(x, y)

    # 修正贝塞尔函数0阶, torch.ops.aten.i0
    @staticmethod
    def i0(x: T) -> T:
        return ir.i0(x)

    # 指数缩放修正贝塞尔函数0阶, torch.special.i0e
    @staticmethod
    def i0e(x: T) -> T:
        return ir.i0e(x)

    # 修正贝塞尔函数1阶, torch.special.i1
    @staticmethod
    def i1(x: T) -> T:
        return ir.i1(x)

    # 指数缩放修正贝塞尔函数1阶, torch.special.i1e
    @staticmethod
    def i1e(x: T) -> T:
        return ir.i1e(x)

    # 下不完全伽马函数, torch.ops.aten.igamma
    @staticmethod
    def igamma(x: T, y: T) -> T:
        return ir.igamma(x, y)

    # 上不完全伽马函数, torch.ops.aten.igammac
    @staticmethod
    def igammac(x: T, y: T) -> T:
        return ir.igammac(x, y)

    # 拉盖尔多项式, torch.special.laguerre_polynomial_l
    @staticmethod
    def laguerre_polynomial_l(x: T, y: T) -> T:
        return ir.laguerre_polynomial_l(x, y)

    # 勒让德多项式, torch.special.legendre_polynomial_p
    @staticmethod
    def legendre_polynomial_p(x: T, y: T) -> T:
        return ir.legendre_polynomial_p(x, y)

    # 对数正态分布CDF, torch.special.log_ndtr
    @staticmethod
    def log_ndtr(x: T) -> T:
        return ir.log_ndtr(x)

    # 修正贝塞尔函数0阶, torch.special.modified_bessel_i0
    @staticmethod
    def modified_bessel_i0(x: T) -> T:
        return ir.modified_bessel_i0(x)

    # 修正贝塞尔函数1阶, torch.special.modified_bessel_i1
    @staticmethod
    def modified_bessel_i1(x: T) -> T:
        return ir.modified_bessel_i1(x)

    # 修正贝塞尔函数K0, torch.special.modified_bessel_k0
    @staticmethod
    def modified_bessel_k0(x: T) -> T:
        return ir.modified_bessel_k0(x)

    # 修正贝塞尔函数K1, torch.special.modified_bessel_k1
    @staticmethod
    def modified_bessel_k1(x: T) -> T:
        return ir.modified_bessel_k1(x)

    # 正态分布CDF, torch.special.ndtr
    @staticmethod
    def ndtr(x: T) -> T:
        return ir.ndtr(x)

    # 正态分布分位数, torch.special.ndtri
    @staticmethod
    def ndtri(x: T) -> T:
        return ir.ndtri(x)

    # 多伽马函数, torch.ops.aten.polygamma
    @staticmethod
    def polygamma(x: T, y: T) -> T:
        return ir.polygamma(x, y)

    # 缩放修正贝塞尔函数K0, torch.special.scaled_modified_bessel_k0
    @staticmethod
    def scaled_modified_bessel_k0(x: T) -> T:
        return ir.scaled_modified_bessel_k0(x)

    # 缩放修正贝塞尔函数K1, torch.special.scaled_modified_bessel_k1
    @staticmethod
    def scaled_modified_bessel_k1(x: T) -> T:
        return ir.scaled_modified_bessel_k1(x)

    # 平移切比雪夫多项式T, torch.special.shifted_chebyshev_polynomial_t
    @staticmethod
    def shifted_chebyshev_polynomial_t(x: T, y: T) -> T:
        return ir.shifted_chebyshev_polynomial_t(x, y)

    # 平移切比雪夫多项式U, torch.special.shifted_chebyshev_polynomial_u
    @staticmethod
    def shifted_chebyshev_polynomial_u(x: T, y: T) -> T:
        return ir.shifted_chebyshev_polynomial_u(x, y)

    # 平移切比雪夫多项式V, torch.special.shifted_chebyshev_polynomial_v
    @staticmethod
    def shifted_chebyshev_polynomial_v(x: T, y: T) -> T:
        return ir.shifted_chebyshev_polynomial_v(x, y)

    # 平移切比雪夫多项式W, torch.special.shifted_chebyshev_polynomial_w
    @staticmethod
    def shifted_chebyshev_polynomial_w(x: T, y: T) -> T:
        return ir.shifted_chebyshev_polynomial_w(x, y)

    # 球面贝塞尔函数0阶, torch.special.spherical_bessel_j0
    @staticmethod
    def spherical_bessel_j0(x: T) -> T:
        return ir.spherical_bessel_j0(x)

    # 黎曼zeta函数, torch.special.zeta
    @staticmethod
    def zeta(x: T, y: T) -> T:
        return ir.zeta(x, y)

    # 逻辑运算类操作7个
    # ----------------------------------------------------------------------------------------------------
    # 逻辑与操作,

    @staticmethod
    def and_(x0: T, x1: T) -> T:
        return ir.and_(x0, x1)

    # 逻辑与, torch.ops.aten.logical_and
    @staticmethod
    def logical_and(x0: T, x1: T) -> T:
        return ir.logical_and(x0, x1)

    # 逻辑非, torch.ops.aten.logical_not
    @staticmethod
    def logical_not(x0: T) -> T:
        return ir.logical_not(x0)

    # 逻辑或, torch.ops.aten.logical_or
    @staticmethod
    def logical_or(x0: T, x1: T) -> T:
        return ir.logical_or(x0, x1)

    # 逻辑异或, torch.ops.aten.logical_xor
    @staticmethod
    def logical_xor(x0: T, x1: T) -> T:
        return ir.logical_xor(x0, x1)

    # 逻辑或操作,
    @staticmethod
    def or_(x0: T, x1: T) -> T:
        return ir.or_(x0, x1)

    # 逻辑异或操作,
    @staticmethod
    def xor(x0: T, x1: T) -> T:
        return ir.xor(x0, x1)

    # 位运算类操作8个
    # ----------------------------------------------------------------------------------------------------
    # 按位与操作, torch.ops.aten.bitwise_and

    @staticmethod
    def bitwise_and(x0: T, x1: T) -> T:
        return ir.bitwise_and(x0, x1)

    # 按位左移, torch.ops.aten.bitwise_left_shift
    @staticmethod
    def bitwise_left_shift(x0: T, x1: T) -> T:
        return ir.bitwise_left_shift(x0, x1)

    # 按位取反, torch.ops.aten.bitwise_not
    @staticmethod
    def bitwise_not(x0: T) -> T:
        return ir.bitwise_not(x0)

    # 按位或操作, torch.ops.aten.bitwise_or
    @staticmethod
    def bitwise_or(x0: T, x1: T) -> T:
        return ir.bitwise_or(x0, x1)

    # 按位右移, torch.ops.aten.bitwise_right_shift
    @staticmethod
    def bitwise_right_shift(x0: T, x1: T) -> T:
        return ir.bitwise_right_shift(x0, x1)

    # 按位异或, torch.ops.aten.bitwise_xor
    @staticmethod
    def bitwise_xor(x0: T, x1: T) -> T:
        return ir.bitwise_xor(x0, x1)

    # 左移操作,
    @staticmethod
    def lshift(x0: T, x1: T) -> T:
        return ir.lshift(x0, x1)

    # 右移操作,
    @staticmethod
    def rshift(x0: T, x1: T) -> T:
        return ir.rshift(x0, x1)

    # 数学函数类操作7个
    # ----------------------------------------------------------------------------------------------------
    # 向上取整, torch.ops.aten.ceil

    @staticmethod
    def ceil(x0: T) -> T:
        return ir.ceil(x0)

    # 向下取整, torch.ops.aten.floor
    @staticmethod
    def floor(x0: T) -> T:
        return ir.floor(x0)

    # 分解浮点数, torch.ops.aten.frexp
    @staticmethod
    def frexp(x0: T):
        return ir.frexp(x0)

    # 欧几里得距离, torch.ops.aten.hypot
    @staticmethod
    def hypot(x0: T, x1: T) -> T:
        return ir.hypot(x0, x1)

    # 四舍五入, torch.ops.aten.round
    @staticmethod
    def round(x0: T) -> T:
        return ir.round(x0)

    # 平方根, torch.ops.aten.sqrt
    @staticmethod
    def sqrt(x0: T) -> T:
        return ir.sqrt(x0)

    # 截断取整, torch.ops.aten.trunc
    @staticmethod
    def trunc(x0: T) -> T:
        return ir.trunc(x0)

    # 类型转换类操作6个
    # ----------------------------------------------------------------------------------------------------
    # 向上取整转整数,

    @staticmethod
    def ceil_to_int(x: T, dtype: 'torch.dtype') -> T:
        return ir.ceil_to_int(x, dtype)

    # 向下取整转整数,
    @staticmethod
    def floor_to_int(x: T, dtype: 'torch.dtype') -> T:
        return ir.floor_to_int(x, dtype)

    # 四舍五入转整数,
    @staticmethod
    def round_to_int(x: T, dtype: 'torch.dtype') -> T:
        return ir.round_to_int(x, dtype)

    # 数据类型转换,
    @staticmethod
    def to_dtype(x: T, dtype: 'torch.dtype', src_dtype: 'Optional[torch.dtype]' = None, use_compute_types: 'bool' = True) -> T:
        if dtype == src_dtype:
            return x
        if dtype == torch.bfloat16:
            dtype = torch.float32
        return ir.cast(x, dst=dtype, src=src_dtype, use_compute_types=use_compute_types)

    # 位转换数据类型,

    @staticmethod
    def to_dtype_bitcast(x: T, dtype: 'torch.dtype', src_dtype: 'torch.dtype') -> T:
        return ir.to_dtype_bitcast(x, dtype, src_dtype)

    # 截断转整数,
    @staticmethod
    def trunc_to_int(x: T, dtype: 'torch.dtype') -> T:
        return ir.trunc_to_int(x, dtype)

    # 边界检查类操作2个
    # ----------------------------------------------------------------------------------------------------
    # 检查边界,

    @staticmethod
    def check_bounds(expr: 'sympy.Expr', size: 'sympy.Expr', lower: 'bool', upper: 'bool') -> 'None':
        return ir.check_bounds(expr, size, lower, upper)

    # Halide边界钳制,
    @staticmethod
    def halide_clamp(value: T, size: 'sympy.Expr', check: 'bool') -> T:
        return ir.halide_clamp(value, size, check)

    # 常量操作类操作1个
    # ----------------------------------------------------------------------------------------------------
    # 定义常量,

    @staticmethod
    def constant(value: 'Union[bool, float, int]', dtype: 'torch.dtype') -> T:
        if dtype == torch.bfloat16:
            dtype = torch.float32
        return ir.constant(repr(value), dtype)

    # 调试工具类操作1个
    # ----------------------------------------------------------------------------------------------------
    # 设备断言,

    @staticmethod
    def device_assert_async(cond: T, msg: 'str') -> T:
        return ir.device_assert_async(cond, msg)

    # 比较运算类操作8个
    # ----------------------------------------------------------------------------------------------------
    # 等于比较, torch.ops.aten.eq

    @staticmethod
    def eq(x0: T, x1: T) -> T:
        return ir.eq(x0, x1)

    # 大于等于比较, torch.ops.aten.ge
    @staticmethod
    def ge(x0: T, x1: T) -> T:
        return ir.ge(x0, x1)

    # 大于比较, torch.ops.aten.gt
    @staticmethod
    def gt(x0: T, x1: T) -> T:
        return ir.gt(x0, x1)

    # 小于等于比较, torch.ops.aten.le
    @staticmethod
    def le(x0: T, x1: T) -> T:
        return ir.le(x0, x1)

    # 小于比较, torch.ops.aten.lt
    @staticmethod
    def lt(x0: T, x1: T) -> T:
        return ir.lt(x0, x1)

    # 最大值, torch.ops.aten.maximum
    @staticmethod
    def maximum(x0: T, x1: T) -> T:
        return ir.maximum(x0, x1)

    # 最小值, torch.ops.aten.minimum
    @staticmethod
    def minimum(x0: T, x1: T) -> T:
        return ir.minimum(x0, x1)

    # 不等于比较, torch.ops.aten.ne
    @staticmethod
    def ne(x0: T, x1: T) -> T:
        return ir.ne(x0, x1)

    # 特殊函数类操作4个
    # ----------------------------------------------------------------------------------------------------
    # 误差函数, torch.ops.aten.erf

    @staticmethod
    def erf(x0: T) -> T:
        return ir.erf(x0)

    # 互补误差函数, torch.ops.aten.erfc
    @staticmethod
    def erfc(x0: T) -> T:
        return ir.erfc(x0)

    # 逆误差函数, torch.ops.aten.erfinv
    @staticmethod
    def erfinv(x0: T) -> T:
        return ir.erfinv(x0)

    # 对数伽马函数, torch.ops.aten.lgamma
    @staticmethod
    def lgamma(x0: T) -> T:
        return ir.lgamma(x0)

    # 指数函数类操作3个
    # ----------------------------------------------------------------------------------------------------
    # 指数函数, torch.ops.aten.exp

    @staticmethod
    def exp(x0: T) -> T:
        return ir.exp(x0)

    # 2的指数函数, torch.ops.aten.exp2
    @staticmethod
    def exp2(x0: T) -> T:
        return ir.exp2(x0)

    # 指数减一, torch.ops.aten.expm1
    @staticmethod
    def expm1(x0: T) -> T:
        return ir.expm1(x0)

    # 基础操作类操作1个
    # ----------------------------------------------------------------------------------------------------
    # 恒等函数,

    @staticmethod
    def identity(x: T) -> T:
        return ir.identity(x)

    # 低级操作类操作1个
    # ----------------------------------------------------------------------------------------------------
    # 内联汇编,

    @staticmethod
    def inline_asm_elementwise(*inputs: T, asm: 'str', constraints: 'Optional[str]' = None, dtype: 'torch.dtype' = torch.float32, is_pure: 'bool' = True, pack: 'int' = 1) -> T:
        return ir.inline_asm_elementwise(*inputs, asm, constraints, dtype, is_pure, pack)

    # 条件判断类操作2个
    # ----------------------------------------------------------------------------------------------------
    # 判断是否为无穷大, torch.ops.aten.isinf

    @staticmethod
    def isinf(x0: T) -> T:
        return ir.isinf(x0)

    # 判断是否为NaN, torch.ops.aten.isnan
    @staticmethod
    def isnan(x0: T) -> T:
        return ir.isnan(x0)

    # 随机数类操作4个
    # ----------------------------------------------------------------------------------------------------
    # 加载随机种子,

    @staticmethod
    def load_seed(name: 'str', offset: T) -> T:
        return ir.load_seed(name, offset)

    # 均匀分布随机数, torch.ops.aten.rand
    @staticmethod
    def rand(seed: T, offset: T) -> T:
        return ir.rand(seed, offset)

    # 整数随机数,
    @staticmethod
    def randint64(seed: T, offset: T, low: T, high: T) -> T:
        return ir.randint64(seed, offset, low, high)

    # 正态分布随机数, torch.ops.aten.randn
    @staticmethod
    def randn(seed: T, offset: T) -> T:
        return ir.randn(seed, offset)

    # 对数函数类操作4个
    # ----------------------------------------------------------------------------------------------------
    # 自然对数, torch.ops.aten.log

    @staticmethod
    def log(x0: T) -> T:
        return ir.log(x0)

    # 常用对数, torch.ops.aten.log10
    @staticmethod
    def log10(x0: T) -> T:
        return ir.log10(x0)

    # 对数, log(1+x), torch.ops.aten.log1p
    @staticmethod
    def log1p(x0: T) -> T:
        return ir.log1p(x0)

    # 以2为底对数, torch.ops.aten.log2
    @staticmethod
    def log2(x0: T) -> T:
        return ir.log2(x0)

    # 控制流类操作2个
    # ----------------------------------------------------------------------------------------------------
    # 掩码操作,

    @staticmethod
    def masked(mask: T, body: 'Callable[[], T]', other: T) -> T:
        return ir.masked(mask, body, other)

    # 条件选择, torch.ops.aten.where
    @staticmethod
    def where(condition: T, x: T, other: T) -> T:
        return ir.where(condition, x, other)

    # 激活函数类操作2个
    # ----------------------------------------------------------------------------------------------------
    # ReLU激活函数, torch.ops.aten.relu

    @staticmethod
    def relu(x0: T) -> T:
        return ir.relu(x0)

    # Sigmoid激活函数, torch.ops.aten.sigmoid
    @staticmethod
    def sigmoid(x0: T) -> T:
        return ir.sigmoid(x0)

    # 符号化表达式
    # ----------------------------------------------------------------------------------------------------
    @staticmethod
    def index_expr(expr: 'sympy.Expr', dtype: 'torch.dtype') -> T:
        return ir.index_expr(expr, dtype)

    # Kernel类操作，这些应该实现到Kernel上去
    # ----------------------------------------------------------------------------------------------------
    @staticmethod
    def load(name: 'str', index: 'sympy.Expr') -> T:
        return ir.load(name, index)

    @staticmethod
    def store(name: 'str', index: 'sympy.Expr', value: T, mode: 'StoreMode' = None) -> 'None':
        return ir.store(name, index, value, mode)

    @staticmethod
    def indirect_indexing(x: T, size: 'sympy.Expr', check: 'bool' = True, wrap_neg=True) -> 'sympy.Expr':
        return ir.indirect_indexing(x, size, check, wrap_neg=True)

    @staticmethod
    def reduction(dtype: 'torch.dtype', src_dtype: 'torch.dtype', reduction_type: 'ReductionType', value: T) -> 'Union[T, tuple[T, ...]]':
        return ir.reduction(dtype, src_dtype, reduction_type, value)

    @staticmethod
    def store_reduction(name: 'str', index: 'sympy.Expr', value: T) -> 'None':
        return ir.store_reduction(name, index, value)

    @staticmethod
    def placeholder(index: 'int') -> T:
        return ir.placeholder(index)

    @staticmethod
    def output(*args: T) -> 'None':
        return ir.output(*args)

    @staticmethod
    def scan(dtypes: 'tuple[torch.dtype, ...]', combine_fn: 'Callable[[tuple[T, ...], tuple[T, ...]], tuple[T, ...]]', values: 'tuple[T, ...]') -> 'tuple[T, ...]':
        return ir.scan(dtypes, combine_fn, values)

    @staticmethod
    def sort(dtypes: 'tuple[torch.dtype, ...]', values: 'tuple[T, ...]', stable: 'bool', descending: 'bool') -> 'tuple[T, ...]':
        return ir.sort(dtypes, values, stable, descending)

    @staticmethod
    def bucketize(values: T, boundaries: 'tuple[str, sympy.Expr, sympy.Expr, sympy.Expr]', boundary_indices: T, indexing_dtype: 'torch.dtype', right: 'bool', sorter: 'Optional[tuple[str, sympy.Expr]]' = None, sorter_indices: 'Optional[T]' = None) -> T:
        return ir.bucketize(values, boundaries, boundary_indices, indexing_dtype, right, sorter, sorter_indices)

    @staticmethod
    def partial_accumulate(
        self,
        name: str,
        reduction_type: ReductionType,
        value: T,
        extra_meta: dict[str, Any],
    ) -> None:
        raise NotImplementedError
