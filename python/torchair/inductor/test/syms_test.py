import unittest

from npu_extension_for_inductor.common.symbols import AscSymbol


class AscSymTest(unittest.TestCase):
    def test_basic(self):
        s0 = AscSymbol('s0')
        s1 = AscSymbol('s1')
        k1 = AscSymbol('1')

        self.assertEqual(repr(s0 * 2), "ascir.SizeExpr([s0,2])")
        self.assertEqual(repr(s0 * s1), "ascir.SizeExpr([s0,s1])")
        self.assertEqual(repr(s0 * s0 * s1), "ascir.SizeExpr([s0,s0,s1])")
        self.assertEqual(repr(s0 * s1 * s0), "ascir.SizeExpr([s0,s1,s0])")
        self.assertEqual(repr(s0 ** 3), "ascir.SizeExpr([s0,s0,s0])")
        self.assertEqual(repr((s0 * s1) ** 3), "ascir.SizeExpr([s0,s1,s0,s1,s0,s1])")

        self.assertEqual(repr(k1), "ascir.SizeExpr([])")
        self.assertEqual(repr(k1 * s1), "ascir.SizeExpr([s1])")
        self.assertEqual(repr(k1 * k1 * s1), "ascir.SizeExpr([s1])")
        self.assertEqual(repr(k1 * s0 * s1), "ascir.SizeExpr([s0,s1])")

    def test_current_unsupported(self):
        s0 = AscSymbol('s0')
        s1 = AscSymbol('s1')
        self.assertEqual(repr(s0 + s1), "ascir.SizeExpr([s0]) + ascir.SizeExpr([s1])")
        self.assertEqual(repr(s0 - s1), "ascir.SizeExpr([s0]) - ascir.SizeExpr([s1])")
        self.assertEqual(repr(s0 / s1), "ascir.SizeExpr([s0])/ascir.SizeExpr([s1])")
        self.assertEqual(repr(s0 // s1), "floor(ascir.SizeExpr([s0])/ascir.SizeExpr([s1]))")


if __name__ == '__main__':
    unittest.main()
