import ast
import os
import shutil
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

# 兜底：如果 user 没显式调 _stub_debugging_host_only，至少通过 env var 让
# inductor_npu_ext 进 cpu 模式。这两条必须在 import inductor_npu_ext 之前设。
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")
os.environ.setdefault("TORCHINDUCTOR_NPU_EXT_DEBUG", "cpu")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch
from torch._inductor import config
import inductor_npu_ext

# 老分支提供的 host-only stub 入口，没有就走 env var 路径。
if hasattr(inductor_npu_ext, "_stub_debugging_host_only"):
    inductor_npu_ext._stub_debugging_host_only()


# ---- asc_graph.py 解析工具 ----
# 落盘格式形如：
#   load1 = ascir.ops.Load('graph0_hint/load1', graph0_hint)
#   load1.x = data1.y
#   load1.y.axis = [p1, p0, p2, p3]
#   load1.y.size = [64, 32, 5, 1]
#   load1.y.strides = [320, 5, 1, 0]
# AST 抓 op 类型 + 输入绑定 + .y.size / .y.strides / .y.axis。

_BINARY_OPS = {"Add", "Sub", "Mul", "Div", "TrueDiv", "FloorDiv", "Mod", "Maximum",
               "Minimum", "Pow", "Ge", "Gt", "Le", "Lt", "Eq", "Ne",
               "BitwiseAnd", "BitwiseOr", "BitwiseXor", "LogicalAnd", "LogicalOr"}


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


def _collect_asc_graphs() -> List[Tuple[Path, Dict[str, dict]]]:
    root = Path.cwd() / "torch_compile_debug"
    if not root.exists():
        return []
    return [(p, _parse_asc_graph(p)) for p in sorted(root.rglob("asc_graph.py"))]


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
                if info["type"] not in _BINARY_OPS:
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

    def test_add(self):
        # Test that compilation and execution do not raise any exceptions
        try:
            @torch.compile
            def func(x, y):
                return x + y

            x = torch.randn(2)
            y = torch.randn(2)
            func(x, y)
        except Exception as e:
            self.fail(f"test_add raised an exception: {e}")

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
        self.assertIn("tiling_def, host_impl, device_impl = fuser.codegen(", e2e_section, "e2e mode should open asc_graph.py")

        default_section = content[content.find("else:"):]
        self.assertIn("tiling_def", default_section, "Default mode should have tiling_def")
        self.assertIn("host_impl", default_section, "Default mode should have host_impl")
        self.assertIn("device_impl", default_section, "Default mode should have device_impl")
        self.assertNotIn("tiling_def, host_impl, device_impl = fuser.codegen(", default_section, "default mode should not open asc_graph.py")

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


if __name__ == "__main__":
    unittest.main()
