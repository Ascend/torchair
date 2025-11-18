import os
import re
import sys
import unittest
import subprocess
import tempfile
import contextlib
import dataclasses
import torch
import torch.nn as nn
import torch._dynamo.config

from torchair.inference._cache_compiler import CompiledModel, ModelCacheSaver


EXPECTED_FILE_PATTERNS_GE = [
    "model__{id}/dynamo_out_graph.txt",
    # forward
    "model__{id}/forward/000_aot_forward_graph.txt",
    "model__{id}/forward/001_aot_forward_graph_after_post_grad_custom_pre_pass.txt",
    "model__{id}/forward/002_aot_forward_graph_after_optimize_noop_ops.txt",
    "model__{id}/forward/003_aot_forward_graph_after_recover_view_inplace_pattern.txt",
    "model__{id}/forward/004_aot_forward_graph_after_optimize_sym_input.txt",
    "model__{id}/forward/005_aot_forward_graph_after_view_to_reshape.txt",
    "model__{id}/forward/006_aot_forward_graph_after_post_grad_custom_post_pass.txt",
    "model__{id}/forward/007_aot_forward_original_ge_graph.pbtxt",
    "model__{id}/forward/008_aot_forward_graph_after_optimize_sym_pack.pbtxt",
    "model__{id}/forward/009_aot_forward_graph_after_remove_dead_data_and_reorder_data_index.pbtxt",
    "model__{id}/forward/010_aot_forward_graph_after_optimize_reference_op_redundant_copy.pbtxt",
    "model__{id}/forward/011_aot_forward_graph_after_mapping_assign_op_to_graph_output.pbtxt",
    "model__{id}/forward/012_aot_forward_graph_after_explicit_order_for_side_effect_nodes.pbtxt",
    "model__{id}/forward/013_aot_forward_graph_after_explicit_order_for_cmo.pbtxt",
    "model__{id}/forward/014_aot_forward_optimized_ge_graph.pbtxt",
    # backward
    "model__{id}/backward/000_aot_backward_graph.txt",
    "model__{id}/backward/001_aot_backward_graph_after_post_grad_custom_pre_pass.txt",
    "model__{id}/backward/002_aot_backward_graph_after_optimize_noop_ops.txt",
    "model__{id}/backward/003_aot_backward_graph_after_recover_view_inplace_pattern.txt",
    "model__{id}/backward/004_aot_backward_graph_after_optimize_sym_input.txt",
    "model__{id}/backward/005_aot_backward_graph_after_view_to_reshape.txt",
    "model__{id}/backward/006_aot_backward_graph_after_post_grad_custom_post_pass.txt",
    "model__{id}/backward/007_aot_backward_original_ge_graph.pbtxt",
    "model__{id}/backward/008_aot_backward_graph_after_optimize_sym_pack.pbtxt",
    "model__{id}/backward/009_aot_backward_graph_after_remove_dead_data_and_reorder_data_index.pbtxt",
    "model__{id}/backward/010_aot_backward_graph_after_optimize_reference_op_redundant_copy.pbtxt",
    "model__{id}/backward/011_aot_backward_graph_after_mapping_assign_op_to_graph_output.pbtxt",
    "model__{id}/backward/012_aot_backward_graph_after_explicit_order_for_side_effect_nodes.pbtxt",
    "model__{id}/backward/013_aot_backward_graph_after_explicit_order_for_cmo.pbtxt",
    "model__{id}/backward/014_aot_backward_optimized_ge_graph.pbtxt",
]

EXPECTED_FILE_PATTERNS_CACHE = [
    "model__{id}/dynamo_out_graph.txt",
    # forward
    "model__{id}/forward/000_aot_forward_graph.txt",
    "model__{id}/forward/001_aot_forward_graph_after_post_grad_custom_pre_pass.txt",
    "model__{id}/forward/002_aot_forward_graph_after_optimize_noop_ops.txt",
    "model__{id}/forward/003_aot_forward_graph_after_recover_view_inplace_pattern.txt",
    "model__{id}/forward/004_aot_forward_graph_after_optimize_sym_input.txt",
    "model__{id}/forward/005_aot_forward_graph_after_view_to_reshape.txt",
    "model__{id}/forward/006_aot_forward_graph_after_post_grad_custom_post_pass.txt",
    "model__{id}/forward/007_aot_forward_original_ge_graph.pbtxt",
    "model__{id}/forward/008_aot_forward_graph_after_optimize_sym_pack.pbtxt",
    "model__{id}/forward/009_aot_forward_graph_after_remove_dead_data_and_reorder_data_index.pbtxt",
    "model__{id}/forward/010_aot_forward_graph_after_optimize_reference_op_redundant_copy.pbtxt",
    "model__{id}/forward/011_aot_forward_graph_after_mapping_assign_op_to_graph_output.pbtxt",
    "model__{id}/forward/012_aot_forward_graph_after_explicit_order_for_side_effect_nodes.pbtxt",
    "model__{id}/forward/013_aot_forward_graph_after_explicit_order_for_cmo.pbtxt",
    "model__{id}/forward/014_aot_forward_optimized_ge_graph.pbtxt",
]


TORCHAIR_PY_PATTERNS = [
    r"(?i)fx2ge_converter.py",
    r"(?i)npu_fx_compiler.py",
    r"(?i)utils.py"
]
TORCHAIR_CPP_PATTERNS = [
    r"(?i)concrete_graph/session.cpp",
    r"(?i)concrete_graph.cpp",
    r"(?i)executor.cpp"
]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def check_logs(pattern_list: list, logs: str) -> bool:
    for pattern in pattern_list:
        if re.search(pattern, logs):
            return True
    return False


def check_torchair_directory_structure(base_dir: str, file_list: list) -> list:
    missing_files = []
    for rel_path in file_list:
        abs_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(abs_path):
            missing_files.append(abs_path)
    return missing_files


def add_sample():
    import torchair
    from torch._dynamo.utils import get_debug_dir
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)
    model = Model()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    print(f"get_debug_dir(): {get_debug_dir()}")
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    model(x, y)


def add_sample_with_backward_and_post_grad_custom():
    import torchair
    from torch._dynamo.utils import get_debug_dir
    
    config = torchair.CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    config.experimental_config.remove_noop_ops = True

    def _custom_pre_fn(gm, example_inputs, compile_config: torchair.CompilerConfig):
        return None

    def _custom_post_fn(gm, example_inputs, compile_config: torchair.CompilerConfig):
        return None
    
    config.post_grad_custom_pre_pass = _custom_pre_fn
    config.post_grad_custom_post_pass = _custom_post_fn
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + x

    model = Model()
    print(f"get_debug_dir(): {get_debug_dir()}")
    device = 'cpu'
    compiled_model = torch.compile(model, backend=npu_backend)
    compiled_model.to(device)
    x = torch.randn(2, 2, requires_grad=True, device=device)
    out = compiled_model(x)
    loss_fn = nn.MSELoss()
    target = torch.randn(2, 2, device=device)
    loss = loss_fn(out, target)
    loss.backward()
    x = torch.randn(2, 2, requires_grad=True, device=device, dtype=torch.float64)
    out = compiled_model(x)
    target = torch.randn(2, 2, device=device, dtype=torch.float64)
    loss = loss_fn(out, target)
    loss.backward()


@dataclasses.dataclass
class InputMeta:
    data: torch.Tensor
    is_prompt: bool


def add_sample_cache():
    import torchair
    from torch._dynamo.utils import get_debug_dir
    config = torchair.CompilerConfig()
    config.experimental_config.remove_noop_ops = True
    config.debug.graph_dump.type = "pbtxt"
    
    def _custom_pre_fn(gm, example_inputs, compile_config: torchair.CompilerConfig):
        return None

    def _custom_post_fn(gm, example_inputs, compile_config: torchair.CompilerConfig):
        return None
    
    config.post_grad_custom_pre_pass = _custom_pre_fn
    config.post_grad_custom_post_pass = _custom_post_fn

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(2, 1)
            self.linear2 = torch.nn.Linear(2, 1)
            for param in self.parameters():
                torch.nn.init.ones_(param)

            self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config)
            self.cached_decode = torchair.inference.cache_compile(self.decode, config=config)

        def forward(self, x, y):
            if x.is_prompt:
                return self.cached_prompt(x, y)
            return self.cached_decode(x, y)

        def _forward(self, x, y):
            return self.linear2(x.data) + self.linear2(y[0])

        def prompt(self, x, y):
            return self._forward(x, y)

        def decode(self, x, y):
            return self._forward(x, y)

    model = Model()
    print(f"get_debug_dir(): {get_debug_dir()}")
    prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt, config=config)
    decode_cache_dir = CompiledModel.get_cache_bin(model.decode, config=config)
    ModelCacheSaver.remove_cache(prompt_cache_dir)
    ModelCacheSaver.remove_cache(decode_cache_dir)

    prompt_data = torch.ones(3, 2, requires_grad=True)
    y_data = torch.ones(3, 2, requires_grad=True)
    prompt1 = InputMeta(prompt_data, True), [y_data]
    
    model(*prompt1)  # first enter CacheBackend


class TestLogDebug(unittest.TestCase):
    _exit_stack: contextlib.ExitStack

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._exit_stack = contextlib.ExitStack()

        tmpdir = cls._exit_stack.enter_context(
            tempfile.TemporaryDirectory(prefix="torchair_debug_")
        )
        cls.DEBUG_DIR = tmpdir  
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch(debug_dir_root=cls.DEBUG_DIR)
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._exit_stack.close()
        cls.DEBUG_DIR = None

    def test_torch_compile_ge_debug_is_1(self):
        self.assertIsNotNone(self.DEBUG_DIR)
        launcher = (
            f"import os; import os,sys; sys.path.insert(0, {SCRIPT_DIR!r}); os.environ['TORCH_COMPILE_DEBUG'] = '1'; "
            "import torchair_debug_log_st as m; m.add_sample_with_backward_and_post_grad_custom()"
        )
        res = subprocess.run(
            [sys.executable, "-u", "-c", launcher],
            cwd=self.DEBUG_DIR,   
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode != 0:
            print(f"debug_dump_subprocess_error: stdout:\n{res.stdout}\nstderr:\n{res.stderr}")

        debug_dir_output = None
        match = re.search(r"get_debug_dir\(\): \s*(\S+)", res.stdout)
        if match:
            debug_dir_output = match.group(1)

        self.assertIsNotNone(debug_dir_output, msg=f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")

        # 1. Verify the existence of the torchair directory
        torchair_root = os.path.join(debug_dir_output, "torchair")
        self.assertTrue(os.path.exists(torchair_root), msg=f"torchair directory does not exist: {torchair_root}")

        # 2. Verify all expected files exist
        expected_files = []
        for model_id in [0, 1]:
            for template in EXPECTED_FILE_PATTERNS_GE:
                rel_path = template.format(id=model_id)
                expected_files.append(rel_path)
        missing_files = check_torchair_directory_structure(torchair_root, expected_files)

        self.assertFalse(missing_files, msg=f"Missing files: {', '.join(missing_files)}")
            
        # Check file count to ensure no extra files
        expected_count = len(EXPECTED_FILE_PATTERNS_GE) * 2
        actual_count = 0
        for root, _, files in os.walk(torchair_root):
            actual_count += len(files)
        self.assertEqual(
            actual_count,
            expected_count,
            msg=f"File count mismatch: expected {expected_count} files, got {actual_count} files"
        )

    def test_torch_cache_compile_debug_is_1(self):
        self.assertIsNotNone(self.DEBUG_DIR)
        launcher = (
            f"import os; import os,sys; sys.path.insert(0, {SCRIPT_DIR!r}); os.environ['TORCH_COMPILE_DEBUG'] = '1'; "
            "import torchair_debug_log_st as m; m.add_sample_cache()"
        )
        res = subprocess.run(
            [sys.executable, "-u", "-c", launcher],
            cwd=self.DEBUG_DIR,   
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        debug_dir_output = None
        match = re.search(r"get_debug_dir\(\): \s*(\S+)", res.stdout)
        if match:
            debug_dir_output = match.group(1)

        self.assertIsNotNone(debug_dir_output, msg=f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")

        # 1. Verify the existence of the torchair directory
        torchair_root = os.path.join(debug_dir_output, "torchair")
        self.assertTrue(os.path.exists(torchair_root), msg=f"torchair directory does not exist: {torchair_root}")

        # 2. Verify all expected files exist
        expected_files = []
        for template in EXPECTED_FILE_PATTERNS_CACHE:
            rel_path = template.format(id=0)
            expected_files.append(rel_path)
        missing_files = check_torchair_directory_structure(torchair_root, expected_files)
        self.assertFalse(missing_files, msg=f"Missing files: {', '.join(missing_files)}")
        expected_count = len(EXPECTED_FILE_PATTERNS_CACHE)
        actual_count = 0
        for root, _, files in os.walk(torchair_root):
            actual_count += len(files)
        self.assertEqual(
            actual_count,
            expected_count,
            msg=f"File count mismatch: expected {expected_count} files, got {actual_count} files"
        )


    def test_no_input_torch_compile_debug(self):
        self.assertIsNotNone(self.DEBUG_DIR)
        launcher = (
            f"import sys; sys.path.insert(0, {SCRIPT_DIR!r}); "
            "import torchair_debug_log_st as m; m.add_sample()"
        )
        res = subprocess.run(
            [sys.executable, "-u", "-c", launcher],
            cwd=self.DEBUG_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        debug_dir_output = None
        match = re.search(r"get_debug_dir\(\): \s*(\S+)", res.stdout)
        if match:
            debug_dir_output = match.group(1)

        if debug_dir_output is None:
            expected_torchair_dir = os.path.join(self.DEBUG_DIR, "torch_compile_debug", "torchair")
            self.assertFalse(os.path.exists(expected_torchair_dir))
        else:
            torchair_dir = os.path.join(debug_dir_output, "torchair")
            self.assertFalse(os.path.exists(torchair_dir))

    def test_torch_compile_debug_is_true(self):
        self.assertIsNotNone(self.DEBUG_DIR)
        launcher = (
            f"import os,sys; sys.path.insert(0, {SCRIPT_DIR!r}); "
            "os.environ['TORCH_COMPILE_DEBUG'] = 'True'; "
            "import torchair_debug_log_st as m; m.add_sample()"
        )
        res = subprocess.run(
            [sys.executable, "-u", "-c", launcher],
            cwd=self.DEBUG_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        debug_dir_output = None
        match = re.search(r"get_debug_dir\(\): \s*(\S+)", res.stdout)
        if match:
            debug_dir_output = match.group(1)

        if debug_dir_output is None:
            expected_torchair_dir = os.path.join(self.DEBUG_DIR, "torch_compile_debug", "torchair")
            self.assertFalse(os.path.exists(expected_torchair_dir))
        else:
            torchair_dir = os.path.join(debug_dir_output, "torchair")
            self.assertFalse(os.path.exists(torchair_dir))


if __name__ == "__main__":
    unittest.main(verbosity=2)
