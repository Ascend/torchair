import os
import shutil
import unittest
from pathlib import Path

os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.environ["TORCHINDUCTOR_NPU_EXT_CACHE_DIR"] = str(Path.cwd() / ".npu_kernels_test")

import torch
import inductor_npu_ext


class TestTaskQueueEnable(unittest.TestCase):
    """Test TASK_QUEUE_ENABLE environment variable behavior."""

    def tearDown(self) -> None:
        for name in ("torch_compile_debug", ".npu_kernels_test"):
            p = Path.cwd() / name
            if p.exists():
                shutil.rmtree(p)

    def test_config_value_matches_env(self):
        from inductor_npu_ext import config

        task_queue_mode = int(os.getenv("TASK_QUEUE_ENABLE", "1"))
        ascend_launch_blocking = os.getenv("ASCEND_LAUNCH_BLOCKING", "0") == "1"
        expected_mode = 0 if ascend_launch_blocking else task_queue_mode

        self.assertEqual(
            config._enable_taskqueue_mode,
            expected_mode,
            f"Config mismatch: expected {expected_mode}, got {config._enable_taskqueue_mode}",
        )

    def test_mode_0_wrapper_codegen(self):
        from inductor_npu_ext import config

        if config._enable_taskqueue_mode != 0:
            self.skipTest("TASK_QUEUE_ENABLE != 0, skipping Mode 0 test")

        @torch.compile
        def test_add_sum(x, y):
            return torch.add(x, y).sum()

        x = torch.randn(32, 1024, dtype=torch.float32).npu()
        y = torch.randn(1, 1024, dtype=torch.float32).npu()
        out = test_add_sum(x, y)
        golden = (x + y).sum()
        max_diff = torch.max(torch.abs(out - golden))
        self.assertLess(max_diff.item(), 1e-3, f"Precision check failed: diff={max_diff.item()}")

        wrapper_paths = list(Path.cwd().rglob("inductor_wrapper.cpp"))
        self.assertGreater(len(wrapper_paths), 0, "Should generate inductor_wrapper.cpp")

        wrapper_content = wrapper_paths[0].read_text(encoding="utf-8", errors="replace")
        self.assertNotIn("MallocWorkspace", wrapper_content, "Mode 0 should use MallocWorkspace")
        self.assertIn("RunOpApiV2", wrapper_content, "Mode 0 should not use RunOpApiV2")

    def test_mode_1_wrapper_codegen(self):
        from inductor_npu_ext import config

        if config._enable_taskqueue_mode != 1:
            self.skipTest("TASK_QUEUE_ENABLE != 1, skipping Mode 1 test")

        @torch.compile
        def test_add_sum(x, y):
            return torch.add(x, y).sum()

        x = torch.randn(32, 1024, dtype=torch.float32).npu()
        y = torch.randn(1, 1024, dtype=torch.float32).npu()
        out = test_add_sum(x, y)
        golden = (x + y).sum()
        max_diff = torch.max(torch.abs(out - golden))
        self.assertLess(max_diff.item(), 1e-3, f"Precision check failed: diff={max_diff.item()}")

        wrapper_paths = list(Path.cwd().rglob("inductor_wrapper.cpp"))
        self.assertGreater(len(wrapper_paths), 0, "Should generate inductor_wrapper.cpp")

        wrapper_content = wrapper_paths[0].read_text(encoding="utf-8", errors="replace")
        self.assertIn("RunOpApiV2", wrapper_content, "Mode 1 should use RunOpApiV2")
        self.assertIn("AllocateWorkspaceTensor", wrapper_content, "Mode 1 should use AllocateWorkspaceTensor")

    def test_mode_2_wrapper_codegen(self):
        from inductor_npu_ext import config

        if config._enable_taskqueue_mode != 2:
            self.skipTest("TASK_QUEUE_ENABLE != 2, skipping Mode 2 test")

        @torch.compile
        def test_add_sum(x, y):
            return torch.add(x, y).sum()

        x = torch.randn(32, 1024, dtype=torch.float32).npu()
        y = torch.randn(1, 1024, dtype=torch.float32).npu()
        out = test_add_sum(x, y)
        golden = (x + y).sum()
        max_diff = torch.max(torch.abs(out - golden))
        self.assertLess(max_diff.item(), 1e-3, f"Precision check failed: diff={max_diff.item()}")

        wrapper_paths = list(Path.cwd().rglob("inductor_wrapper.cpp"))
        self.assertGreater(len(wrapper_paths), 0, "Should generate inductor_wrapper.cpp")

        wrapper_content = wrapper_paths[0].read_text(encoding="utf-8", errors="replace")
        self.assertIn("RunOpApiV2", wrapper_content, "Mode 2 should use RunOpApiV2")
        self.assertIn("AllocateWorkspaceTensor", wrapper_content, "Mode 2 should use AllocateWorkspaceTensor")

    def test_ascend_launch_blocking_override(self):
        from inductor_npu_ext import config

        if os.getenv("ASCEND_LAUNCH_BLOCKING", "0") != "1":
            self.skipTest("ASCEND_LAUNCH_BLOCKING != 1, skipping override test")

        self.assertEqual(config._enable_taskqueue_mode, 0, "ASCEND_LAUNCH_BLOCKING=1 should force mode to 0")

        @torch.compile
        def test_add_sum(x, y):
            return torch.add(x, y).sum()

        x = torch.randn(32, 1024, dtype=torch.float32).npu()
        y = torch.randn(1, 1024, dtype=torch.float32).npu()
        out = test_add_sum(x, y)
        golden = (x + y).sum()
        max_diff = torch.max(torch.abs(out - golden))
        self.assertLess(max_diff.item(), 1e-3, f"Precision check failed: diff={max_diff.item()}")

        wrapper_paths = list(Path.cwd().rglob("inductor_wrapper.cpp"))
        self.assertGreater(len(wrapper_paths), 0, "Should generate inductor_wrapper.cpp")

        wrapper_content = wrapper_paths[0].read_text(encoding="utf-8", errors="replace")
        self.assertIn("RunOpApiV2", wrapper_content, "ASCEND_LAUNCH_BLOCKING=1 should not use RunOpApiV2")


if __name__ == "__main__":
    unittest.main()
