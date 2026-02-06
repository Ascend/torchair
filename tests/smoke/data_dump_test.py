import logging
import unittest
import time
from pathlib import Path

import torch
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger


logger.setLevel(logging.DEBUG)


class DataDumpTest(unittest.TestCase):

    def test_data_dump_with_scope(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, data0, data1):
                add_01 = torch.add(data0, data1)
                with torchair.scope.data_dump():
                    sub_01 = torch.sub(data0, data1)
                return add_01, sub_01

        input0 = torch.randn(2, 2, dtype=torch.float16).npu()
        input1 = torch.randn(2, 2, dtype=torch.float16).npu()
        config = torchair.CompilerConfig()
        config.dump_config.enable_dump = True
        config.dump_config.dump_layer = " Add "
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_mode = Network()
        npu_mode = torch.compile(npu_mode, backend=npu_backend)
        npu_mode(input0, input1)
        torch.npu.synchronize()
        cwd = Path.cwd()
        rank_dirs = sorted(cwd.glob("worldsize*_global_rank*"))
        assert rank_dirs, "No rank directory found for dump validation"
        rank_dir = rank_dirs[0]
        pattern = "*/0/graph_*/1/0"
        candidates = sorted(rank_dir.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0)
        assert candidates, f"No dump subdirectories found under {rank_dir} with pattern {pattern}"
        target_dir = candidates[-1]

        files = [p.name for p in target_dir.iterdir() if p.is_file()]
        assert files, f"No files found in dump directory {target_dir}"
        files_lower = [n.lower() for n in files]
        has_add = any("add" in n for n in files_lower)
        has_sub = any("sub" in n for n in files_lower)
        assert has_add and has_sub, f"Dump files missing expected ops: add={has_add}, sub={has_sub}; files={files}"


    def test_aclgraph_data_dump(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, data0, data1):
                add_01 = torch.add(data0, data1)
                sub_01 = torch.sub(data0, data1)
                return add_01, sub_01

        input0 = torch.randn(2, 2, dtype=torch.float16).npu()
        input1 = torch.randn(2, 2, dtype=torch.float16).npu()
        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        config.dump_config.enable_dump = True
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_mode = Network()
        npu_mode = torch.compile(npu_mode, backend=npu_backend)
        npu_mode(input0, input1)
        torch.npu.synchronize()
        from pathlib import Path
        cwd = Path.cwd()
        rank_dirs = sorted(cwd.glob("worldsize*_global_rank*"))
        assert rank_dirs, "No rank directory found for dump validation"
        rank_dir = rank_dirs[0]

        files = [p.name for p in rank_dir.iterdir() if p.is_file()]
        assert files, f"No files found in dump directory {rank_dir}"
        files_lower = [n.lower() for n in files]
        has_add = any("add" in n for n in files_lower)
        has_sub = any("sub" in n for n in files_lower)
        assert has_add and has_sub, f"Dump files missing expected ops: add={has_add}, sub={has_sub}; files={files}"


    def test_aclgraph_data_dump_has_saved(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, data0, data1):
                mm_01 = torch.mm(data0, data1)
                torch_npu.save_npugraph_tensor(mm_01, save_path="./test.pt")
                sq1 = torch.square(mm_01)
                return mm_01, sq1

        input0 = torch.randn(2, 2, dtype=torch.float16).npu()
        input1 = torch.randn(2, 2, dtype=torch.float16).npu()
        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        config.dump_config.enable_dump = True
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_mode = Network()
        npu_mode = torch.compile(npu_mode, backend=npu_backend)
        npu_mode(input0, input1)
        torch.npu.synchronize()
        from pathlib import Path
        cwd = Path.cwd()
        rank_dirs = sorted(cwd.glob("worldsize*_global_rank*"))
        assert rank_dirs, "No rank directory found for dump validation"
        rank_dir = rank_dirs[0]

        files = [p.name for p in rank_dir.iterdir() if p.is_file()]
        assert files, f"No files found in dump directory {rank_dir}"
        files_lower = [n.lower() for n in files]
        has_mm = any("mm" in n for n in files_lower)
        has_pow = any("pow" in n for n in files_lower)
        assert (not has_mm) and has_pow, f"Dump files missing expected ops: mm={has_mm}, sub={has_pow}; files={files}"


if __name__ == '__main__':
    unittest.main()