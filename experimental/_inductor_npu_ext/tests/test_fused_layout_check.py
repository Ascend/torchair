# pylint: disable=R0401
import os
import shutil
import unittest
from pathlib import Path

os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

import torch
import inductor_npu_ext  # noqa: F401 pylint: disable=R0401


class TestFusedLayoutCheckSmoke(unittest.TestCase):
    """Assert fused inductor wrappers mention maybe_check_fused_input_layout in output_code.py."""

    def tearDown(self) -> None:
        for name in ("torch_compile_debug", ".npu_kernels_root"):
            p = Path.cwd() / name
            if p.exists():
                shutil.rmtree(p)

    def test_output_code_contains_maybe_check_fused_input_layout(self):
        # Same compute as README.md (add then sum); NPU tensors; numeric check + output_code scan.
        @torch.compile
        def test_add_sum(x, y):
            return torch.add(x, y).sum()

        x = torch.randn(32, 1024, dtype=torch.float32).npu()
        y = torch.randn(1, 1024, dtype=torch.float32).npu()
        out = test_add_sum(x, y)
        golden = (x + y).sum()
        max_diff = torch.max(torch.abs(out - golden))
        self.assertLess(max_diff.item(), 1e-3, max_diff)

        debug_root = Path.cwd() / "torch_compile_debug"
        output_paths = list(debug_root.rglob("output_code.py"))
        self.assertGreater(
            len(output_paths),
            0,
            f"Expected output_code.py under {debug_root} (TORCH_COMPILE_DEBUG=1).",
        )
        combined = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in output_paths)
        self.assertIn(
            "maybe_check_fused_input_layout",
            combined,
            "Generated wrapper should call maybe_check_fused_input_layout for fused inputs.",
        )

    def test_output_code_contains_maybe_check_fused_input_layout_dynamic(self):
        # Alternates two static shapes in a loop to trigger extra compile paths; still expect
        # maybe_check_fused_input_layout in emitted output_code.py.
        @torch.compile
        def test_add_sum(x, y):
            x = x + y
            y = x + y
            return torch.add(x, y).sum()

        x = torch.randn(32, 1024, dtype=torch.float32).npu()
        y = torch.randn(1, 1024, dtype=torch.float32).npu()

        xx = torch.randn(20, 4068, dtype=torch.float32).npu()
        yy = torch.randn(1, 4068, dtype=torch.float32).npu()
        for i in range(4):
            if i % 2 == 0:
                _ = test_add_sum(x, y)  # noqa: F841
            else:
                _ = test_add_sum(xx, yy)  # noqa: F841

        debug_root = Path.cwd() / "torch_compile_debug"
        output_paths = list(debug_root.rglob("output_code.py"))
        self.assertGreater(
            len(output_paths),
            0,
            f"Expected output_code.py under {debug_root} (TORCH_COMPILE_DEBUG=1).",
        )
        combined = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in output_paths)
        self.assertIn(
            "maybe_check_fused_input_layout",
            combined,
            "Generated wrapper should call maybe_check_fused_input_layout for fused inputs.",
        )


if __name__ == "__main__":
    unittest.main()
