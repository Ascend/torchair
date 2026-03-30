import shutil
import unittest
from pathlib import Path

import torch
from torch._inductor import config
import inductor_npu_ext

inductor_npu_ext._stub_debugging_host_only()


class TestInductorNpuExt(unittest.TestCase):
    def tearDown(self) -> None:
        torch_compile_debug = Path.cwd() / "torch_compile_debug"
        npu_kernels_root = Path.cwd() / ".npu_kernels_root"

        if torch_compile_debug.exists():
            shutil.rmtree(torch_compile_debug)
        if npu_kernels_root.exists():
            shutil.rmtree(npu_kernels_root)

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


if __name__ == "__main__":
    unittest.main()
