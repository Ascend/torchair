import ast
import os
import shutil
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

# 兜底：如果 user 没显式调 _stub_debugging_host_only，至少通过 env var 让
# inductor_npu_ext 进 cpu 模式。这两条必须在 import inductor_npu_ext 之前设。
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")
os.environ.setdefault("TORCHINDUCTOR_NPU_EXT_DEBUG", "cpu+decompose+lowering")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch._inductor import config

import inductor_npu_ext

# AscIR op 分类：
# - SHAPE_CHANGING：reduce/broadcast/transpose 等，输出 shape 允许跟输入不同。
# - ELEMENTWISE_BINARY：x1.size 必须等于 x2.size。
# - ELEMENTWISE_UNARY：x.size 必须等于 y.size。
SHAPE_CHANGING = {
    "Sum", "Max", "Min", "Mean", "Prod", "Argmax", "Argmin",
    "Broadcast", "Transpose", "Reshape", "Squeeze", "Unsqueeze",
    "Data", "Load", "Store", "Output", "Workspace", "Scalar",
    "IndirectLoad", "IndirectIndexing", "CheckBounds",
}
ELEMENTWISE_BINARY = {
    "Mul", "Add", "Sub", "TrueDiv", "Div", "FloorDiv", "Mod", "FMod",
    "Pow", "Maximum", "Minimum", "Eq", "Ne", "Lt", "Gt", "Le", "Ge",
    "BitwiseAnd", "BitwiseOr", "LogicalAnd", "LogicalOr", "Where",
    "Copysign", "Nextafter", "Remainder",
}
ELEMENTWISE_UNARY = {
    "Cast", "Abs", "Exp", "Sqrt", "Rsqrt", "Sigmoid", "Sign",
    "Log", "Ln", "Neg", "Reciprocal", "Square", "Silu", "Floor",
    "Ceil", "Round", "Trunc", "BitwiseNot", "LogicalNot", "Signbit",
    "Sin", "Cos", "Tan", "Tanh", "Asin", "Acos", "Atan",
}


def _eval_int_list(node):
    if isinstance(node, ast.List):
        out = []
        for e in node.elts:
            if isinstance(e, ast.Constant):
                out.append(e.value)
            elif isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) and isinstance(e.operand, ast.Constant):
                out.append(-e.operand.value)
            elif isinstance(e, ast.Name):
                out.append(e.id)
            else:
                out.append(ast.unparse(e))
        return out
    return None


def _parse_asc_graph(path: Path) -> Dict[str, dict]:
    src = path.read_text()
    tree = ast.parse(src)
    ops: Dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        # 形态 1: name = ascir.ops.OpType(...)
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
            func = node.value.func
            if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "ops"):
                ops[target.id] = {"type": func.attr, "inputs": {}, "size": None, "strides": None, "axis": None}
            continue
        # 形态 2: name.attr.attr... = expr
        if not isinstance(target, ast.Attribute):
            continue
        chain = []
        cur = target
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            continue
        op_name = cur.id
        chain.reverse()
        if op_name not in ops:
            continue
        # name.x / name.x1 / name.x2 / name.x3 = something.y
        if len(chain) == 1 and chain[0] in ("x", "x1", "x2", "x3"):
            val = node.value
            if isinstance(val, ast.Attribute) and val.attr == "y" and isinstance(val.value, ast.Name):
                ops[op_name]["inputs"][chain[0]] = val.value.id
            continue
        if len(chain) == 2 and chain[0] == "y":
            if chain[1] == "size":
                ops[op_name]["size"] = _eval_int_list(node.value)
            elif chain[1] == "strides":
                ops[op_name]["strides"] = _eval_int_list(node.value)
            elif chain[1] == "axis":
                ops[op_name]["axis"] = _eval_int_list(node.value)
    return ops

def _check_consistency(ops: Dict[str, dict]) -> List[str]:
    """检查 binary op 两路输入 size 是否一致、unary elementwise op 输入输出 size 是否一致。"""
    issues: List[str] = []
    for name, info in ops.items():
        op_type = info["type"]
        inputs = info["inputs"]

        if op_type in ELEMENTWISE_BINARY and len(inputs) >= 2:
            sized: List[Tuple[str, str, list]] = []
            for slot, src_name in inputs.items():
                src = ops.get(src_name)
                if src and src["size"] is not None:
                    sized.append((slot, src_name, src["size"]))
            if len(sized) >= 2:
                base = tuple(sized[0][2])
                for slot, src_name, size in sized[1:]:
                    if tuple(size) != base:
                        issues.append(
                            f"{name}({op_type}) 输入 shape 不一致: "
                            f"{sized[0][0]}<-{sized[0][1]} size={list(base)} vs "
                            f"{slot}<-{src_name} size={list(size)}"
                        )
                        break

        if op_type in ELEMENTWISE_UNARY and "x" in inputs:
            src = ops.get(inputs["x"])
            out = info["size"]
            if src and src["size"] is not None and out is not None:
                if tuple(src["size"]) != tuple(out):
                    issues.append(
                        f"{name}({op_type}) 输入输出 shape 跳变: "
                        f"x<-{inputs['x']} size={src['size']} -> y size={out}"
                    )
    return issues


def _collect_asc_graphs() -> List[Tuple[Path, Dict[str, dict]]]:
    """扫 torch_compile_debug 下所有 asc_graph.py 并解析。"""
    debug_root = Path.cwd() / "torch_compile_debug"
    if not debug_root.exists():
        return []
    paths = sorted(debug_root.rglob("asc_graph.py"))
    return [(p, _parse_asc_graph(p)) for p in paths]


def _contig_stride(size):
    """跟 DenseLoop.contiguous_stride 对齐：size=1 维 stride=0；其余维 stride=
    product(后面所有维 size)。"""
    stride = []
    mult = 1
    for s in reversed(size):
        if isinstance(s, int) and s == 1:
            stride.append(0)
        else:
            stride.append(mult)
        if isinstance(s, int):
            mult *= s
    stride.reverse()
    return stride


class TestInductorNpuExt(unittest.TestCase):
    def setUp(self) -> None:
        import torch._dynamo
        torch._dynamo.reset()
        config.trace.enabled = True
        # 防止上次落盘的 asc_graph 串到本用例
        for d in (Path.cwd() / "torch_compile_debug", Path.cwd() / ".npu_kernels_root"):
            if d.exists():
                shutil.rmtree(d)

    def tearDown(self) -> None:
        torch_compile_debug = Path.cwd() / "torch_compile_debug"
        npu_kernels_root = Path.cwd() / ".npu_kernels_root"

        if torch_compile_debug.exists():
            shutil.rmtree(torch_compile_debug)
        if npu_kernels_root.exists():
            shutil.rmtree(npu_kernels_root)

    # ---- view 推导校验工具 ----

    def _assert_contig_non_load(self, graphs):
        """除 Load 外的所有节点：stride 必须 == contiguous_stride(size)。"""
        issues = []
        for path, ops in graphs:
            for name, info in ops.items():
                if info["type"] in ("Load", "Data", "Output", "Workspace", "Scalar"):
                    continue
                size, stride = info["size"], info["strides"]
                if size is None or stride is None:
                    continue
                expect = _contig_stride(size)
                if list(stride) != list(expect):
                    issues.append(f"[{path.parent.name}] {name}({info['type']}) "
                                  f"size={size} stride={stride} != contig {expect}")
        self.assertEqual(issues, [], "non-Load nodes 必须 contiguous_stride:\n  "
                         + "\n  ".join(issues))

    def _assert_binary_axis_consistent(self, graphs):
        """二元 op 的 x1/x2 输入 axis 必须完全一致（含顺序）。"""
        issues = []
        for path, ops in graphs:
            for name, info in ops.items():
                if info["type"] not in ELEMENTWISE_BINARY:
                    continue
                src_axes = []
                for slot in ("x1", "x2"):
                    src_name = info["inputs"].get(slot)
                    if src_name and src_name in ops and ops[src_name]["axis"] is not None:
                        src_axes.append((slot, src_name, ops[src_name]["axis"]))
                if len(src_axes) >= 2 and src_axes[0][2] != src_axes[1][2]:
                    issues.append(f"[{path.parent.name}] {name}({info['type']}) "
                                  f"两路输入 axis 不一致: "
                                  f"{src_axes[0][0]}<-{src_axes[0][1]} {src_axes[0][2]} vs "
                                  f"{src_axes[1][0]}<-{src_axes[1][1]} {src_axes[1][2]}")
        self.assertEqual(issues, [], "二元 op 输入 axis 必须一致:\n  "
                         + "\n  ".join(issues))

    def _assert_has_op(self, graphs, op_type, hint=""):
        for _, ops in graphs:
            if any(o["type"] == op_type for o in ops.values()):
                return
        kinds = [(p.parent.name, sorted({o["type"] for o in ops.values()})) for p, ops in graphs]
        self.fail(f"asc_graph 未找到 {op_type} 节点 ({hint})；现有 kernels: {kinds}")

    def _run_and_collect(self, func, *args):
        with torch.no_grad():
            func(*args)
        graphs = _collect_asc_graphs()
        self.assertGreater(len(graphs), 0, "未生成任何 asc_graph.py")
        return graphs

    def _assert_fused_in_one_graph(self, graphs, required_ops, hint=""):
        """断言 required_ops 中所有类型在同一张 asc_graph 中出现。"""
        for path, ops in graphs:
            present = {o["type"] for o in ops.values()}
            if all(t in present for t in required_ops):
                return path, ops
        kernels = [(p.parent.name, sorted({o["type"] for o in ops.values()}))
                   for p, ops in graphs]
        self.fail(f"未找到同时包含 {required_ops} 的 kernel ({hint})；"
                  f"现有 kernels: {kernels}")

    def _assert_all_consistent(self, graphs):
        all_issues = []
        for path, ops in graphs:
            for issue in _check_consistency(ops):
                all_issues.append(f"[{path.parent.name}] {issue}")
        self.assertEqual(all_issues, [], "asc_graph 形状不一致:\n  " + "\n  ".join(all_issues))

    def test_benchmark_generation(self):

        @torch.compile
        def func(x, y):
            return x + y

        config.trace.enabled = True

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        func(x, y)

        benchmark_files = list(Path.cwd().rglob("benchmark.py"))
        self.assertGreater(len(benchmark_files), 0, "Should generate benchmark.py")

        benchmark_path = benchmark_files[0]
        content = benchmark_path.read_text()

        required_elements = [
            "import sys",
            "import torch",
            "import torch_npu",
            "async_compile_ascendc",
            "torch_npu.profiler.profile",
            "if __name__ == '__main__':",
            "if sys.argv[-1] == 'e2e':",
            "else:",
        ]

        for element in required_elements:
            self.assertIn(element, content, f"benchmark.py should contain {element}")

        e2e_section = content[content.find("if sys.argv[-1] == 'e2e':"):content.find("else:")]
        self.assertIn("tiling_def, host_impl, device_impl = fuser.codegen(",
                      e2e_section, "e2e mode should open asc_graph.py")

        default_section = content[content.find("else:"):]
        self.assertIn("tiling_def", default_section, "Default mode should have tiling_def")
        self.assertIn("host_impl", default_section, "Default mode should have host_impl")
        self.assertIn("device_impl", default_section, "Default mode should have device_impl")
        self.assertNotIn("tiling_def, host_impl, device_impl = fuser.codegen(",
                         default_section, "default mode should not open asc_graph.py")

        benchmark_path.unlink()

    # ---- view 组合用例 ----
    # 共同断言：
    #   (1) 除 Load 外所有节点必须满足 stride == contiguous_stride(size)（ascir 约束）
    #   (2) 二元 op 两路输入 axis 必须完全一致（曾经 _get_view_road bug 导致不一致）
    # 各用例额外断言：图中出现了对应的 view op（Broadcast / Transpose）。

    def test_view_pure_broadcast(self):
        """单维 1→N broadcast，没 transpose。"""
        @torch.compile
        def fn(x, y):
            return x + y

        graphs = self._run_and_collect(
            fn,
            torch.randn(1, 8, 16),  # broadcast 到 [4,8,16]
            torch.randn(4, 8, 16),
        )
        self._assert_has_op(graphs, "Broadcast", "pure_broadcast")
        self._assert_contig_non_load(graphs)
        self._assert_binary_axis_consistent(graphs)

    def test_view_multi_dim_broadcast(self):
        """多维 1→N broadcast 一起出现。"""
        @torch.compile
        def fn(x, y):
            return x * y

        graphs = self._run_and_collect(
            fn,
            torch.randn(1, 1, 5),      # broadcast 到 [3,4,5]，2 维同时升
            torch.randn(3, 4, 5),
        )
        self._assert_has_op(graphs, "Broadcast", "multi_dim_broadcast")
        self._assert_contig_non_load(graphs)
        self._assert_binary_axis_consistent(graphs)

    def test_view_pure_transpose(self):
        """单纯 permute 让 src.axis 跟 dst.axis 顺序不同（无 broadcast）。"""
        @torch.compile
        def fn(x, y):
            return x.permute(1, 0, 2) + y

        graphs = self._run_and_collect(
            fn,
            torch.randn(4, 8, 16),
            torch.randn(8, 4, 16),
        )
        self._assert_has_op(graphs, "Transpose", "pure_transpose")
        self._assert_contig_non_load(graphs)
        self._assert_binary_axis_consistent(graphs)

    def test_view_transpose_then_broadcast(self):
        """main21 同款：permute + broadcast 一次完成 —— 历史上漏 broadcast、
        且 transpose 输出 stride 不连续，是这个用例要兜住的回归点。"""
        @torch.compile
        def fn(x, y):
            return y * x.permute(1, 0, 2, 3)

        graphs = self._run_and_collect(
            fn,
            torch.randn(64, 32, 5, 1),       # permute 后 [32,64,5,1]，最后 broadcast 到 [32,64,5,56]
            torch.randn(32, 64, 5, 56),
        )
        self._assert_has_op(graphs, "Transpose", "transpose_then_broadcast")
        self._assert_has_op(graphs, "Broadcast", "transpose_then_broadcast")
        self._assert_contig_non_load(graphs)
        self._assert_binary_axis_consistent(graphs)

    def test_view_unsqueeze_broadcast(self):
        """unsqueeze（隐式插 size=1 维）+ broadcast：跟 transpose 路径不同，
        但同样要保证 Broadcast 节点 contig。"""
        @torch.compile
        def fn(x, y):
            return x.unsqueeze(-1) + y

        graphs = self._run_and_collect(
            fn,
            torch.randn(4, 8),           # → [4, 8, 1] → broadcast [4, 8, 16]
            torch.randn(4, 8, 16),
        )
        self._assert_has_op(graphs, "Broadcast", "unsqueeze_broadcast")
        self._assert_contig_non_load(graphs)
        self._assert_binary_axis_consistent(graphs)

    # ---- 白名单算子看护 ----
    # 看护 aten_lowering.py 开放的 lowering 白名单算子，确保它们都能成功转换为
    # 合法的 ascir graph（不退化回 eager）。每个 case 独立 reset + 清 debug 目录。
    # 共同断言：
    #   - 至少生成一个 asc_graph.py（说明走了 NPU 后端，没整体 fallback）
    #   - 除 Load 外节点 stride == contiguous_stride(size)
    #   - 二元 op 两路输入 axis 一致
    #   - 给定 expect_op 时，断言对应 ascir op 真出现在图里（确认目标算子真 lower）

    def _lower_and_check(self, fn, args, expect_op=None):
        import torch._dynamo
        torch._dynamo.reset()
        for d in (Path.cwd() / "torch_compile_debug", Path.cwd() / ".npu_kernels_root"):
            if d.exists():
                shutil.rmtree(d)
        with torch.no_grad():
            torch.compile(fn)(*args)
        graphs = _collect_asc_graphs()
        self.assertGreater(len(graphs), 0, "未生成任何 asc_graph.py")
        if expect_op is not None:
            self._assert_has_op(graphs, expect_op)
        self._assert_contig_non_load(graphs)
        self._assert_binary_axis_consistent(graphs)

    def test_lowering_pointwise(self):
        """白名单 pointwise 算子各自能 lower 成对应 ascir op。"""
        r = torch.randn
        # (expect_ascir_op, fn, args)。expect=None 表示算子会被 inductor 拆成多个
        # 原子 op（silu/remainder），只校验图合法不锚定单一 op。
        cases = [
            ("Add",     lambda a, b: a + b,                         [r(8, 16), r(8, 16)]),
            ("Sub",     lambda a, b: a - b,                         [r(8, 16), r(8, 16)]),
            ("Mul",     lambda a, b: a * b,                         [r(8, 16), r(8, 16)]),
            ("TrueDiv", lambda a, b: a / (b.abs() + 1.0),           [r(8, 16), r(8, 16)]),
            # 张量指数，避免 inductor 把 x**const 优化成连乘（那样图里只剩 Mul）
            ("Pow",     lambda a, b: a.abs() ** (b.abs() + 1.0),    [r(8, 16), r(8, 16)]),
            ("Sqrt",    lambda a: torch.sqrt(a.abs() + 1.0),        [r(8, 16)]),
            ("Rsqrt",   lambda a: torch.rsqrt(a.abs() + 1.0),       [r(8, 16)]),
            ("Abs",     lambda a: a.abs() + 1.0,                    [r(8, 16)]),
            ("Exp",     lambda a: torch.exp(a * 0.1),               [r(8, 16)]),
            ("Sigmoid", lambda a: torch.sigmoid(a),                 [r(8, 16)]),
            ("Relu",    lambda a: torch.relu(a) + 1.0,              [r(8, 16)]),
            ("Neg",     lambda a: -a + 1.0,                         [r(8, 16)]),
            ("Sign",    lambda a: torch.sgn(a) + 1.0,               [r(8, 16)]),
            ("Log1p",   lambda a: torch.log1p(a.abs() + 1.0),       [r(8, 16)]),
            (None,      lambda a: torch.nn.functional.silu(a),      [r(8, 16)]),
            (None,      lambda a, b: torch.remainder(a, b.abs() + 1.0), [r(8, 16), r(8, 16)]),
            # floor_divide：inductor 靠 decomposition 拆成 div+floor（需 _finetune_decompose
            # 保留它的 decomposition），无单一 ascir op
            (None,      lambda a, b: torch.floor_divide(a.abs(), b.abs() + 1.0), [r(8, 16), r(8, 16)]),
        ]
        for expect_op, fn, args in cases:
            with self.subTest(op=expect_op or "compound"):
                self._lower_and_check(fn, args, expect_op)

    def test_lowering_compare(self):
        """比较 op 输出 bool —— 看护 support_out_dtypes 放行 bool/uint8、且
        convert_element_type 接受 bool 输入两处配置，缺一会 fallback 回 eager。
        用 .to(float32) 把 bool 转回，模拟典型用法。"""
        r = torch.randn
        cases = [
            ("Ge", lambda a, b: (a >= b).to(torch.float32), [r(8, 16), r(8, 16)]),
            ("Le", lambda a, b: (a <= b).to(torch.float32), [r(8, 16), r(8, 16)]),
            ("Gt", lambda a, b: (a > b).to(torch.float32),  [r(8, 16), r(8, 16)]),
            ("Lt", lambda a, b: (a < b).to(torch.float32),  [r(8, 16), r(8, 16)]),
            ("Eq", lambda a, b: (a == b).to(torch.float32), [r(8, 16), r(8, 16)]),
            ("Ne", lambda a, b: (a != b).to(torch.float32), [r(8, 16), r(8, 16)]),
        ]
        for expect_op, fn, args in cases:
            with self.subTest(op=expect_op):
                self._lower_and_check(fn, args, expect_op)

    def test_lowering_reduce_and_convert(self):
        """白名单 reduce（sum/mean/max/min）+ dtype 转换能 lower。"""
        r = torch.randn
        cases = [
            ("Sum",  lambda a: a.sum(-1),                  [r(8, 16)]),
            ("Sum",  lambda a: a.mean(-1),                 [r(8, 16)]),  # mean = sum * (1/n)
            ("Max",  lambda a: torch.max(a),               [r(8, 16)]),
            ("Min",  lambda a: torch.min(a),               [r(8, 16)]),
            ("Cast", lambda a: (a + 1.0).to(torch.float16), [r(8, 16)]),
        ]
        for expect_op, fn, args in cases:
            with self.subTest(op=expect_op):
                self._lower_and_check(fn, args, expect_op)

    def test_lowering_view_ops(self):
        """白名单 view 类算子（permute/unsqueeze/squeeze/select/slice/expand/
        transpose）必须跟计算 op 组合才会产生实际融合；只校验生成的图合法
        （view 可能被 reinterpret 进 load 的 stride，不强求出现独立 view 节点）。"""
        r = torch.randn
        cases = [
            ("permute",   lambda a, b: a.permute(1, 0) + b,   [r(8, 16), r(16, 8)]),
            ("unsqueeze", lambda a, b: a.unsqueeze(-1) + b,   [r(8, 16), r(8, 16, 4)]),
            ("squeeze",   lambda a, b: a.squeeze(1) + b,      [r(8, 1, 16), r(8, 16)]),
            ("select",    lambda a, b: a.select(0, 0) + b,    [r(4, 8, 16), r(8, 16)]),
            ("slice",     lambda a, b: a[:, 1:3] + b,         [r(8, 16), r(8, 2)]),
            ("expand",    lambda a, b: a.expand(8, 16) * b,   [r(1, 16), r(8, 16)]),
            ("transpose", lambda a, b: a.transpose(0, 1) + b, [r(8, 16), r(16, 8)]),
        ]
        for name, fn, args in cases:
            with self.subTest(view=name):
                self._lower_and_check(fn, args, expect_op=None)

    def test_soc_gating(self):
        """看护 _LoweringGuard.support(since=...) 的 SoC gating 逻辑。

        gating 规则：current_soc 已知且 since 给定且 current_soc < since 时
        跳过注册（该算子在当前 SoC 上会 fallback）；否则注册。

        UT 跑在 cpu 模式（current_soc=None），gating 默认不生效，所以这里直接
        patch lowering.common.current_soc 模拟各档 SoC，用白名单外的探针 op
        （aten.atan）验证注册与否，避免污染真实白名单、也不依赖真实设备。
        """
        from inductor_npu_ext.lowering import common as lc
        from inductor_npu_ext.lowering.common import float_dtypes
        from inductor_npu_ext.common import Soc

        probe = torch.ops.aten.atan.default  # 不在白名单里
        saved = lc.current_soc

        def reg(soc, since):
            lc.current_soc = soc
            lc._LoweringGuard._data.pop(probe, None)
            lc._LoweringGuard.support(probe, float_dtypes(), since=since)
            return lc._LoweringGuard.has(probe)

        try:
            # current_soc < since → 跳过注册（在该 SoC 上 fallback）
            self.assertFalse(reg(Soc.A2, Soc.A5), "A2 < A5 应跳过 since=A5 注册")
            self.assertFalse(reg(Soc.A3, Soc.A5), "A3 < A5 应跳过")
            # current_soc >= since → 注册
            self.assertTrue(reg(Soc.A5, Soc.A5), "A5 >= A5 应注册")
            # since=None → 不做 gating，恒注册
            self.assertTrue(reg(Soc.A2, None), "since=None 不 gating，应注册")
            # current_soc=None（cpu/nothrow 调试模式）→ 不做 gating，恒注册
            self.assertTrue(reg(None, Soc.A5), "current_soc=None 不 gating，应注册")
        finally:
            lc.current_soc = saved
            lc._LoweringGuard._data.pop(probe, None)  # 清理探针，勿污染其它用例

    def test_axis_merge_reduce_then_cast(self):
        """sum(dim=-1) → cast 到 fp16；cast 是 1D 节点，应跟 2D Sum 融到同一 kernel。"""
        @torch.compile
        def func(x):
            return x.sum(dim=-1).to(torch.float16)

        graphs = self._run_and_collect(func, torch.randn(8, 32, dtype=torch.float32))
        self._assert_fused_in_one_graph(graphs, ["Sum", "Cast"], "reduce_then_cast")
        self._assert_all_consistent(graphs)

    def test_axis_merge_reduce_then_pointwise(self):
        """sum(dim=-1) → cast → mul → add；多个 1D pointwise 串在 2D reduce 后。"""
        @torch.compile
        def func(x):
            return (x.sum(dim=-1).to(torch.float16) * 2 + 1)

        graphs = self._run_and_collect(func, torch.randn(8, 32, dtype=torch.float32))
        self._assert_fused_in_one_graph(graphs, ["Sum", "Mul", "Add"],
                                        "reduce_then_pointwise")
        self._assert_all_consistent(graphs)

    def test_axis_merge_mean(self):
        """mean = sum / N，post-reduce 上有 1D 除法节点。"""
        @torch.compile
        def func(x):
            return x.mean(dim=-1)

        graphs = self._run_and_collect(func, torch.randn(8, 32, dtype=torch.float32))
        # 必含 Sum；除法可能被 inductor 折成乘以 1/N，所以放宽不强制 TrueDiv。
        self._assert_fused_in_one_graph(graphs, ["Sum"], "mean")
        self._assert_all_consistent(graphs)

    # --- softmax / log_softmax：两次 reduce + 中间 broadcast，非常典型 ---
    def test_axis_merge_softmax(self):
        """softmax: Max → Sub → Exp → Sum → TrueDiv，理想下融到一个 kernel 里。"""
        @torch.compile
        def func(x):
            return torch.softmax(x, dim=-1)

        graphs = self._run_and_collect(func, torch.randn(8, 64, dtype=torch.float32))
        self._assert_fused_in_one_graph(graphs, ["Max", "Sum", "Exp", "TrueDiv"],
                                        "softmax")
        self._assert_all_consistent(graphs)

    def test_axis_merge_log_softmax(self):
        """log_softmax: Max + Sub + Exp + Sum + Log + Sub。"""
        @torch.compile
        def func(x):
            return torch.log_softmax(x, dim=-1)

        graphs = self._run_and_collect(func, torch.randn(8, 64, dtype=torch.float32))
        self._assert_fused_in_one_graph(graphs, ["Max", "Sum", "Exp"], "log_softmax")
        self._assert_all_consistent(graphs)

    # --- 水平融合：两个独立 reduce 进同一个 kernel ---
    def test_axis_merge_sum_horizontal_fusion(self):
        """两个独立 input 的 sum + 后续 add，scheduler 水平融合。"""
        @torch.compile
        def func(x, y):
            return x.sum(dim=-1) + y.sum(dim=-1)

        graphs = self._run_and_collect(func,
                                       torch.randn(16, 32, dtype=torch.float32),
                                       torch.randn(16, 32, dtype=torch.float32))
        # 期望两个 Sum 都在同一个 kernel；count 一下。
        path, ops = self._assert_fused_in_one_graph(graphs, ["Sum", "Add"],
                                                    "sum_horizontal_fusion")
        sum_count = sum(1 for o in ops.values() if o["type"] == "Sum")
        self.assertGreaterEqual(sum_count, 2,
                                f"水平融合预期 ≥2 个 Sum，实际 {sum_count}（{path.parent.name}）")
        self._assert_all_consistent(graphs)

    # --- 水平融合2：两个独立 reduce 进同一个 kernel ---
    def test_axis_merge_sum_horizontal_fusion2(self):
        """两个独立 input 的 sum + 后续 add，scheduler 水平融合。"""
        @torch.compile
        def func(x):
            return x.abs().sum(dim=-1), x.mul(3).sum(dim=-1)

        graphs = self._run_and_collect(func,
                                       torch.randn(16, 32, dtype=torch.float32))
        # 期望两个 Sum 都在同一个 kernel；count 一下。
        path, ops = self._assert_fused_in_one_graph(graphs, ["Sum", "Abs", "Mul"],
                                                    "sum_horizontal_fusion")
        sum_count = sum(1 for o in ops.values() if o["type"] == "Sum")
        self.assertGreaterEqual(sum_count, 2,
                                f"水平融合预期 ≥2 个 Sum，实际 {sum_count}（{path.parent.name}）")
        self._assert_all_consistent(graphs)

    def test_axis_merge_sum_two_outputs(self):
        """两个 sum 都作为输出返回。"""
        @torch.compile
        def func(x, y):
            return x.sum(dim=-1), y.sum(dim=-1)

        graphs = self._run_and_collect(func,
                                       torch.randn(16, 32, dtype=torch.float32),
                                       torch.randn(16, 32, dtype=torch.float32))
        # 两个独立 sum 没共同消费者，scheduler 不一定融合，只校验一致性。
        self._assert_all_consistent(graphs)

if __name__ == "__main__":
    unittest.main()
