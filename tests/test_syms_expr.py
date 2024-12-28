import unittest
import sympy

from npu_extension_for_inductor.common.symbols import AscExpr


class AscSymTest(unittest.TestCase):
    def assert_repr_eq(self, a, b):
        self.assertEqual(repr(AscExpr(a)), b)

    def assert_expand_repr_eq(self, a, b):
        self.assertEqual(repr(AscExpr(a).expand_pow()), b)

    def test_basic(self):
        s0 = sympy.Symbol('s0')
        s1 = sympy.Symbol('s1')
        s2 = sympy.Symbol('s2')
        s3 = sympy.Symbol('s3')
        s4 = sympy.Symbol('s4')
        k1 = 1

        self.assert_repr_eq(s0*2, "ascir.SizeExpr(2)*s0")
        self.assert_repr_eq(s0*s1, "s0*s1")
        self.assert_repr_eq(s0*s0*s1, "s0**ascir.SizeExpr(2)*s1")
        self.assert_repr_eq(s0*s1*s0, "s0**ascir.SizeExpr(2)*s1")
        self.assert_repr_eq(s0 ** 3, "s0**ascir.SizeExpr(3)")
        self.assert_repr_eq((s0*s1) ** 3, "s0**ascir.SizeExpr(3)*s1**ascir.SizeExpr(3)")

        self.assert_repr_eq(s0 + 3, "ascir.SizeExpr(3) + s0")
        self.assert_repr_eq(3 + s0, "ascir.SizeExpr(3) + s0")
        self.assert_repr_eq(3 - s0, "-ascir.SizeExpr(3) + s0")
        self.assert_repr_eq(-s0, "-s0")
        self.assert_repr_eq(-2, "ascir.SizeExpr(-2)")
        self.assert_repr_eq(s0 - 3, "-ascir.SizeExpr(3) + s0")
        self.assert_repr_eq(s0 / -3, "-s0/ascir.SizeExpr(3)")
        self.assert_repr_eq(-3 / s0, "ascir.SizeExpr(-3)/s0")

        self.assert_repr_eq(k1, "ascir.SizeExpr(1)")
        self.assert_repr_eq(k1*s1, "s1")
        self.assert_repr_eq(k1*k1*s1, "s1")
        self.assert_repr_eq(k1*s0*s1, "s0*s1")

        self.assert_repr_eq(s0 + s1, "s0 + s1")
        self.assert_repr_eq(s0 - s1, "s0 - s1")
        self.assert_repr_eq(s0 / s1, "s0/s1")

        self.assert_repr_eq(s0 + s1 + s0, "ascir.SizeExpr(2)*s0 + s1")
        self.assert_repr_eq(s0 - s1 - s0, "-s1")
        self.assert_repr_eq(s0 / s1 / s0, "ascir.SizeExpr(1)/s1")

        self.assert_repr_eq(s0 + s1 - s2 * s3 / s4, "s0 + s1 - s2*s3/s4")
        self.assert_repr_eq(s0 + 2 * s1 - s2 * s3 * 3 / s4 / 4,
                            "ascir.SizeExpr(2)*s1 - ascir.SizeExpr(3)*s2*s3/(ascir.SizeExpr(4)*s4) + s0")

        self.assert_expand_repr_eq(s0*2, "ascir.SizeExpr(2)*s0")
        self.assert_expand_repr_eq(s0*s0*s1, "s0*s0*s1")
        self.assert_expand_repr_eq(s0*s1*s0, "s0*s0*s1")
        self.assert_expand_repr_eq(s0 ** 3, "s0*s0*s0")
        self.assert_expand_repr_eq((s0*s1) ** 3, "s0*s0*s0*s1*s1*s1")

    def test_current_unsupported(self):
        s0 = sympy.Symbol('s0')
        s1 = sympy.Symbol('s1')
        self.assert_repr_eq(s0 // s1, "floor(s0/s1)")


if __name__ == '__main__':
    unittest.main()
