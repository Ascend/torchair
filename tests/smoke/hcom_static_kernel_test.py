import os
import unittest
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
import torch_npu
import torchair

torch.manual_seed(7)
torch.npu.manual_seed_all(7)


class HcomStaticKernelTest(unittest.TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        torchair.patch_for_hcom()
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29505'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_static_kernel_with_cache(cls, rank, world_size, input, cache_dir):
        class CacheHcomModel(torch.nn.Module):
            def __init__(self, config, cache_dir):
                super(CacheHcomModel, self).__init__()
                self.relu = torch.nn.ReLU()
                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config, cache_dir=cache_dir)

            def prompt(self, x):
                return self._forward(x)

            def forward(self, x):
                return self.cached_prompt(x)

            def _forward(self, x):
                relu_01 = self.relu(x)
                reshape_01 = torch.reshape(relu_01, (1, 32, 1, 128))
                softmax_01 = torch.nn.functional.softmax(reshape_01)
                sqrt_01 = torch.sqrt(softmax_01)
                relu_02 = self.relu(sqrt_01)
                square_01 = torch.square(relu_02)
                torch_npu.distributed.distributed_c10d.dist.all_reduce(square_01)
                add_01 = torch.add(square_01, square_01)
                return add_01
        torch.npu.set_device(rank)
        HcomStaticKernelTest._init_dist_hccl(rank, world_size)

        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        npu_model = CacheHcomModel(config, cache_dir).npu()
        input0 = input.npu()
        npu_output = npu_model(input0)


    @classmethod
    def _test_static_kernel_without_cache(cls, rank, world_size, input, kernel_build_dir):
        class HcomModel(torch.nn.Module):
            def __init__(self):
                super(HcomModel, self).__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                relu_01 = self.relu(x)
                reshape_01 = torch.reshape(relu_01, (1, 32, 1, 128))
                softmax_01 = torch.nn.functional.softmax(reshape_01)
                sqrt_01 = torch.sqrt(softmax_01)
                relu_02 = self.relu(sqrt_01)
                square_01 = torch.square(relu_02)
                torch_npu.distributed.distributed_c10d.dist.all_reduce(square_01)
                add_01 = torch.add(square_01, square_01)
                return add_01
        torch.npu.set_device(rank)
        HcomStaticKernelTest._init_dist_hccl(rank, world_size)
        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = kernel_build_dir
        input0 = input.npu()
        npu_model = HcomModel().npu()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_model = torch.compile(npu_model, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_output = npu_model(input0)

    def test_static_kernel_without_cache(self):
        kernel_build_dir = "./static_kernel_dir_without_cache"
        if os.path.exists(kernel_build_dir):
            shutil.rmtree(kernel_build_dir)
        os.makedirs(kernel_build_dir, exist_ok = True)

        result = os.popen("ls /dev | grep davinci | wc -l")
        dev_num = result.read()
        result.close()
        device_size = int(dev_num) - 1
        if device_size < 2:
            return

        world_size = device_size
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

        with torch.multiprocessing.Manager() as manager:
            input = torch.randn(1, 4, 8, 128, dtype=torch.float16)
            torch.multiprocessing.spawn(HcomStaticKernelTest._test_static_kernel_without_cache,
                                        args=(world_size, input, kernel_build_dir),
                                        nprocs=world_size, join=True)

        static_kernel_dir_path = Path(kernel_build_dir)
        self.assertTrue(static_kernel_dir_path.exists())
        outputs_dirs = [d for d in static_kernel_dir_path.iterdir() if
                        d.is_dir() and d.name == "aclnn_static_shape_kernel_outputs"]
        self.assertEqual(len(outputs_dirs), 1)
        ts_outputs_dirs = [d for d in outputs_dirs[0].iterdir() if
                           d.is_dir() and d.name.endswith("_outputs") and d.name.startswith("ts")]
        self.assertEqual(len(ts_outputs_dirs), world_size)
        run_pkgs = list(outputs_dirs[0].rglob("*.run"))
        self.assertEqual(len(run_pkgs), 1)

    def test_static_kernel_with_cache(self):
        cache_dir = "./static_kernel_dir_with_cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok = True)

        result = os.popen("ls /dev | grep davinci | wc -l")
        dev_num = result.read()
        result.close()
        device_size = int(dev_num) - 1
        if device_size < 2:
            return

        world_size = device_size
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

        with torch.multiprocessing.Manager() as manager:
            input = torch.randn(1, 4, 8, 128, dtype=torch.float16)
            torch.multiprocessing.spawn(HcomStaticKernelTest._test_static_kernel_with_cache,
                                        args=(world_size, input, cache_dir),
                                        nprocs=world_size, join=True)

        cache_dir_path = Path(cache_dir)
        self.assertTrue(cache_dir_path.exists())
        cachemode_dirs = [d for d in cache_dir_path.iterdir() if
                        d.is_dir() and d.name.startswith("CacheHcomModel")]
        self.assertEqual(len(cachemode_dirs), 1)
        cache_rank_dirs = [d for d in cachemode_dirs[0].iterdir() if d.is_dir()]
        self.assertEqual(len(cache_rank_dirs), world_size)
        run_pkgs = list(cachemode_dirs[0].rglob("*.run"))
        self.assertEqual(len(run_pkgs), 1)
        run_pkg_path = run_pkgs[0].resolve()
        rank_0_dir = f'world{world_size}global_rank0'
        self.assertTrue(rank_0_dir in str(run_pkg_path))

        # load cache
        with torch.multiprocessing.Manager() as manager:
            input = torch.randn(1, 4, 8, 128, dtype=torch.float16)
            torch.multiprocessing.spawn(HcomStaticKernelTest._test_static_kernel_with_cache,
                                        args=(world_size, input, cache_dir),
                                        nprocs=world_size, join=True)

        self.assertTrue(cache_dir_path.exists())
        run_pkgs_02 = list(cache_dir_path.rglob("*.run"))
        self.assertEqual(len(run_pkgs), 1)
        self.assertEqual(str(run_pkg_path), str(run_pkgs_02[0].resolve()))

if __name__ == '__main__':
    unittest.main()