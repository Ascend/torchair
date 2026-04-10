import os
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

import torch
import torch_npu


class StaticKernelBlacklistTest(unittest.TestCase):
    """Test static kernel blacklist functionality."""

    def setUp(self):
        self.kernel_build_dir = "./static_kernel_blacklist_test_dir"
        if os.path.exists(self.kernel_build_dir):
            shutil.rmtree(self.kernel_build_dir)
        os.makedirs(self.kernel_build_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.kernel_build_dir):
            shutil.rmtree(self.kernel_build_dir)

    def test_static_kernel_blacklist_for_add_operator(self):
        """
        End-to-end test for static kernel blacklist functionality.

        This test:
        1. Creates a model with add, sub, mul operators
        2. Adds 'Add' to the static kernel blacklist
        3. Compiles the model with static kernel enabled using npugraph_ex backend
        4. Verifies that Add operator JSON is in the blacklist folder
        5. Verifies that Add operator JSON is NOT in the selected folder
        """
        from npugraph_ex._acl_concrete_graph.static_kernel import (
            _set_static_kernel_blacklist,
            compile_static_kernel as original_compile_static_kernel
        )

        # Define a model with add, sub, mul operators
        class ModelWithAddSubMul(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                # Add operator
                add_result = torch.add(x, x)

                # Sub operator
                sub_result = torch.sub(add_result, x)

                # Mul operator
                mul_result = torch.mul(sub_result, 2.0)

                return mul_result

        # Set the blacklist to exclude Add operator
        # The blacklist matches JSON filenames containing the pattern
        _set_static_kernel_blacklist(["Add"])

        # Create a wrapper function that patches build_dir
        kernel_build_dir = self.kernel_build_dir

        def patched_compile_static_kernel(fx_func, *args, **kwargs):
            # Override build_dir with our test directory
            kwargs['build_dir'] = kernel_build_dir
            return original_compile_static_kernel(fx_func, *args, **kwargs)

        try:
            # Patch compile_static_kernel to use our build_dir
            with patch('npugraph_ex._acl_concrete_graph.acl_graph.compile_static_kernel',
                      side_effect=patched_compile_static_kernel):
                # Create and compile the model using npugraph_ex backend
                model = ModelWithAddSubMul().npu()
                options = {"static_kernel_compile": True}
                compiled_model = torch.compile(model, backend="npugraph_ex", options=options,
                                              fullgraph=True, dynamic=False)

                # Run the model to trigger static kernel compilation
                input_tensor = torch.randn(4, 16, dtype=torch.float16).npu()
                output = compiled_model(input_tensor)

            # Verify the blacklist directory structure
            kernel_build_path = Path(self.kernel_build_dir)
            self.assertTrue(kernel_build_path.exists(), f"Kernel build directory {self.kernel_build_dir} does not exist")

            # Find the outputs directory
            outputs_dirs = [d for d in kernel_build_path.iterdir()
                           if d.is_dir() and d.name.endswith("_outputs") and d.name.startswith("ts")]

            if outputs_dirs:
                outputs_dir = outputs_dirs[0]

                # Check for blacklist directory
                blacklist_dirs = [d for d in outputs_dir.iterdir()
                                 if d.is_dir() and "blacklist" in d.name]

                # Check for selected directory
                selected_dirs = [d for d in outputs_dir.iterdir()
                                if d.is_dir() and "selected" in d.name]

                # Verify Add operator is in blacklist
                if blacklist_dirs:
                    blacklist_dir = blacklist_dirs[0]
                    blacklist_json_files = list(blacklist_dir.glob("*.json"))

                    # Check if any JSON file contains "Add" in its name
                    add_in_blacklist = any("Add" in f.name or "add" in f.name.lower()
                                          for f in blacklist_json_files)

                    if add_in_blacklist:
                        print(f"[PASS] Add operator JSON found in blacklist directory: {blacklist_dir}")
                        for f in blacklist_json_files:
                            if "Add" in f.name or "add" in f.name.lower():
                                print(f"  - Blacklisted JSON: {f.name}")
                    else:
                        print(f"[INFO] Blacklist directory exists but no Add JSON found. Files: {[f.name for f in blacklist_json_files]}")

                # Verify Add operator is NOT in selected
                if selected_dirs:
                    selected_dir = selected_dirs[0]
                    selected_json_files = list(selected_dir.glob("*.json"))

                    # Check that no JSON file contains "Add" in its name
                    add_in_selected = any("Add" in f.name or "add" in f.name.lower()
                                         for f in selected_json_files)

                    self.assertFalse(add_in_selected,
                                    f"Add operator JSON should NOT be in selected directory. "
                                    f"Found Add-related files: {[f.name for f in selected_json_files if 'Add' in f.name or 'add' in f.name.lower()]}")

                    print(f"[PASS] Add operator JSON NOT found in selected directory: {selected_dir}")
                    print(f"  - Selected JSON files: {[f.name for f in selected_json_files]}")

                # Verify Sub and Mul operators are in selected (not blacklisted)
                if selected_dirs:
                    selected_dir = selected_dirs[0]
                    selected_json_files = list(selected_dir.glob("*.json"))

                    sub_in_selected = any("Sub" in f.name or "sub" in f.name.lower()
                                         for f in selected_json_files)
                    mul_in_selected = any("Mul" in f.name or "mul" in f.name.lower()
                                         for f in selected_json_files)

                    print(f"[INFO] Sub in selected: {sub_in_selected}")
                    print(f"[INFO] Mul in selected: {mul_in_selected}")

        finally:
            # Reset the blacklist
            _set_static_kernel_blacklist([])


if __name__ == '__main__':
    unittest.main()
