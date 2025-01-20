import os
import unittest
from unittest.mock import Mock, patch
import shutil

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.multiprocessing as mp
import torch_npu
import torchair


class CacheHcomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cached_module = torchair.inference.cache_compile(self.prompt, dynamic=False)

    def inner_forward(self, x, y):
        ret = x + y
        torch.distributed.all_reduce(ret)
        return ret

    def forward(self, x, y):
        return self.cached_module(x, y)

    def prompt(self, x, y):
        return self.inner_forward(x, y)


class HcomCacheTest(unittest.TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        torchair.patch_for_hcom()
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29510'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _init_dist_hccl_without_patch(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29510'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_hccl_cache_not_create_pg(cls, rank, world_size, init_pg):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        unuse_pg = torch.distributed.new_group()
        model = CacheHcomModel().npu()
        x = torch.ones(2, 2).npu()
        y = torch.ones(2, 2).npu()
        mocked_new_group = Mock(side_effect=dist.new_group)
        mocked_find_or_create_pg = Mock(side_effect=torch.distributed.distributed_c10d.\
                                        _find_or_create_pg_by_ranks_and_tag)
        with patch('torch.distributed.new_group') as mocked_new_group, \
             patch('torch.distributed.distributed_c10d._find_or_create_pg_by_ranks_and_tag') as \
                mocked_find_or_create_pg:
            ret = model(x, y)
        assert (mocked_new_group.called == False)
        assert (mocked_find_or_create_pg.call_count == 1) # 只在convert中触发一次
        torch.distributed.destroy_process_group()

    @classmethod
    def _test_hccl_create_cache_get_hccl_comm_name(cls, rank, world_size, init_pg):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        unuse_pg = torch.distributed.new_group()
        model = CacheHcomModel().npu()
        x = torch.ones(2, 2).npu()
        y = torch.ones(2, 2).npu()
        pg_name = c10d._world.default_pg._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        with patch.object(torch_npu._C._distributed_c10d.ProcessGroupHCCL, 'get_hccl_comm_name', \
                          return_value=pg_name) as get_hccl_comm_name:
            ret = model(x, y)
        assert (get_hccl_comm_name.call_count == 4)
        # 第一次是convert中调用
        assert (get_hccl_comm_name.call_args_list[0].kwargs['init_comm'] == True)
        # 第二、三次是torchair编译期为找MC2调用，存在default_pg、unuse_pg, 不会尝试初始化
        assert (get_hccl_comm_name.call_args_list[1].kwargs['init_comm'] == False)
        assert (get_hccl_comm_name.call_args_list[2].kwargs['init_comm'] == False)
        # 第四次是cache时执行codegen代码调用，尝试初始化
        assert (get_hccl_comm_name.call_args_list[3].kwargs['init_comm'] == True)
        dist.destroy_process_group()

    @classmethod
    def _test_hccl_use_cache_get_hccl_comm_name(cls, rank, world_size, init_pg):
        torch.npu.set_device(rank)
        init_pg(rank, world_size)
        unuse_pg = torch.distributed.new_group()
        model = CacheHcomModel().npu()
        x = torch.ones(2, 2).npu()
        y = torch.ones(2, 2).npu()
        pg_name = c10d._world.default_pg._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        with patch.object(torch_npu._C._distributed_c10d.ProcessGroupHCCL, 'get_hccl_comm_name', \
                          return_value=pg_name) as get_hccl_comm_name:
            ret = model(x, y)
        assert (get_hccl_comm_name.call_count == 1)
        # cache时执行codegen代码调用，尝试初始化
        assert (get_hccl_comm_name.call_args_list[0].kwargs['init_comm'] == True)
        dist.destroy_process_group()

    @classmethod
    def check_cache_file_and_clean_env(cls, path: str = ''):
        if not path:
            path = ".torchair_cache"
        assert os.path.exists(path)
        shutil.rmtree(path)

    def _test_multiprocess(self, f, init_pg, world_size):
        ctx = mp.get_context('spawn')
        ps = []
        for rank in range(world_size):
            p = ctx.Process(target=f, args=(rank, world_size, init_pg))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()


    def test_cache_codegen(self):
        ranks = [2]
        for world_size in ranks:
            self._test_multiprocess(HcomCacheTest._test_hccl_cache_not_create_pg,
                                    HcomCacheTest._init_dist_hccl, world_size)
        HcomCacheTest.check_cache_file_and_clean_env()
        for world_size in ranks:
            self._test_multiprocess(HcomCacheTest._test_hccl_create_cache_get_hccl_comm_name,
                                    HcomCacheTest._init_dist_hccl, world_size)
        for world_size in ranks:
            self._test_multiprocess(HcomCacheTest._test_hccl_use_cache_get_hccl_comm_name,
                                    HcomCacheTest._init_dist_hccl, world_size)
        HcomCacheTest.check_cache_file_and_clean_env()

    @unittest.skipIf(torch.__version__ < '2.3.1', "patch needed for torch version < 2.3.1")
    def test_cache_codegen_without_patch(self):
        ranks = [2]
        for world_size in ranks:
            self._test_multiprocess(HcomCacheTest._test_hccl_cache_not_create_pg,
                                    HcomCacheTest._init_dist_hccl_without_patch, world_size)
        HcomCacheTest.check_cache_file_and_clean_env()
        for world_size in ranks:
            self._test_multiprocess(HcomCacheTest._test_hccl_create_cache_get_hccl_comm_name,
                                    HcomCacheTest._init_dist_hccl_without_patch, world_size)
        for world_size in ranks:
            self._test_multiprocess(HcomCacheTest._test_hccl_use_cache_get_hccl_comm_name,
                                    HcomCacheTest._init_dist_hccl_without_patch, world_size)
        HcomCacheTest.check_cache_file_and_clean_env()

if __name__ == '__main__':
    unittest.main()
