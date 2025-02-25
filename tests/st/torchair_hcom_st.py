import unittest
from unittest.mock import Mock, patch
from dataclasses import dataclass

import torch
from torch._inductor.utils import IndentedBuffer
import torchair
from torchair._ge_concrete_graph.hcom_utils import get_group_name_and_record, record_pg_to_graph, \
    rename_cached_pgname, codegen_refresh_cache_pgname, encode_pg_tag_ranklist
from torchair.ge._ge_graph import GeGraph
from torchair._ge_concrete_graph import ge_apis as ge
from torchair.ge._ge_graph import DataType

tag_name = 0
unqiue_pg_name = 0
global_ranklist = None
default_pg = None
pg_map = {}
pg_name_map = {}
pg_name_init_map = []


class MockPG():
    def __init__(self, tag=None, ranklist=None):
        if tag:
            self.tag = tag
        else:
            global tag_name
            tag_name = tag_name + 1
            self.tag = "tag" + str(tag_name)
        self.ranklist = ranklist if ranklist else global_ranklist
        self.is_init_comm = False
        self.pg_name = None

    def get_hccl_comm_name(self, rank, init_comm=True):
        if init_comm:
            self.is_init_comm = True
        if not self.is_init_comm:
            return ''
        else:
            if not self.pg_name:
                if pg_name_map.get(0) is None:
                    global unqiue_pg_name
                    unqiue_pg_name = unqiue_pg_name + 1
                    self.pg_name = "pg_name" + str(unqiue_pg_name)
                else:
                    if pg_name_map.get(0) not in pg_name_init_map:
                        self.pg_name = pg_name_map.get(0)
                    else:
                        raise RuntimeError("The current PG has been used!")
            pg_name_init_map.append(self.pg_name)
            return self.pg_name

    def _set_hccl_comm_name(self, group_name):
        pg_name_map[0] = group_name

    def _get_backend(self, device):
        return self


def patch_init_process_group(backend, world_size, rank):
    global default_pg, global_ranklist, pg_map
    global_ranklist = [0, 1]
    default_pg = MockPG()
    pg_map[default_pg] = (default_pg.tag, default_pg.ranklist)


def patch_new_group():
    pg = MockPG()
    global pg_map
    pg_map[pg] = (pg.tag, pg.ranklist)
    return pg


def patch_find_or_create_pg_by_ranks_and_tag(tag, rank_list, group_size):
    global pg_map
    for key, value in pg_map.items():
        if value == (tag, rank_list):
            return key
    return MockPG()


def patch_find_pg_by_ranks_and_tag(tag, rank_list):
    global pg_map
    for key, value in pg_map.items():
        if value == (tag, rank_list):
            return key
    return None


def patch_get_rank():
    return 0


def patch_get_process_group_ranks(pg):
    return pg.ranklist


def patch_get_group_tag(pg):
    return pg.tag


def patch_is_initialized():
    return True


@dataclass
class DeviceType:
    type = "npu"


def patch_get_pg_default_device(pg):
    device = DeviceType()
    return device


def patch_get_backend(pg):
    return "hccl"


def patch_device(unused_param):
    return ""


class PatchWorld:
    @property
    def pg_map(self):
        global pg_map
        return pg_map


PatchWorld = PatchWorld()

torch.distributed.init_process_group = patch_init_process_group
torch.distributed.new_group = patch_new_group
torch.distributed.distributed_c10d._find_or_create_pg_by_ranks_and_tag = patch_find_or_create_pg_by_ranks_and_tag
torch.distributed.distributed_c10d._find_pg_by_ranks_and_tag = patch_find_pg_by_ranks_and_tag
torch.distributed.get_rank = patch_get_rank
torch.distributed.distributed_c10d.get_process_group_ranks = patch_get_process_group_ranks
torch.distributed.distributed_c10d._get_group_tag = patch_get_group_tag
torch.distributed.is_initialized = patch_is_initialized
torch.distributed.distributed_c10d._get_pg_default_device = patch_get_pg_default_device
torch.distributed.distributed_c10d._world = PatchWorld
torch.distributed.distributed_c10d.get_backend = patch_get_backend
torch.device = patch_device


class TorchairSt(unittest.TestCase):
    def setUp(self) -> None:
        global tag_name, unqiue_pg_name, global_ranklist, default_pg, pg_map
        tag_name = 0
        unqiue_pg_name = 0
        global_ranklist = None
        default_pg = None
        pg_map = {}
        return super().setUp()

    def test_torch_compile(self):
        torch.distributed.init_process_group(
            backend='hccl', world_size=2, rank=0)
        # 模拟convert调用
        self.assertEqual(get_group_name_and_record(
            "tag1", [0, 1], 2), 'pg_name1')
        self.assertTrue(default_pg.is_init_comm)

    def test_torch_compile_with_unuse_pg(self):
        torch.distributed.init_process_group(
            backend='hccl', world_size=2, rank=0)
        unuse_pg = torch.distributed.new_group()
        # 模拟convert调用
        self.assertEqual(get_group_name_and_record(
            "tag1", [0, 1], 2), 'pg_name1')
        self.assertTrue(default_pg.is_init_comm)
        self.assertFalse(unuse_pg.is_init_comm)

    def test_torch_compile_with_mc2(self):
        torch.distributed.init_process_group(
            backend='hccl', world_size=2, rank=0)
        mc2_pg = torch.distributed.new_group()
        self.assertEqual(mc2_pg._get_backend(
            "device_npu").get_hccl_comm_name(0, init_comm=True), "pg_name1")

        # 模拟convert调用,使用allreduce非mc2, default_pg pg先创建，hccl后初始化, 因此pg_name2
        self.assertEqual(get_group_name_and_record(
            "tag1", [0, 1], 2), 'pg_name2')
        self.assertTrue(default_pg.is_init_comm)
        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.HcomAllReduce(x, reduction="sum",
                                 group=default_pg.pg_name, fusion=0)
            z = ge.MatmulAllReduce(x, y, group=mc2_pg.pg_name, bias=None,
                                   x3=None,
                                   antiquant_scale=None,
                                   antiquant_offset=None,
                                   dequant_scale=None,
                                   pertoken_scale=None,
                                   comm_quant_scale_1=None,
                                   comm_quant_scale_2=None,
                                   reduce_op="sum",
                                   is_trans_a=1,
                                   is_trans_b=1,
                                   comm_turn=1,
                                   antiquant_group_size=1)
        # 记录MC2
        record_pg_to_graph(graph)
        self.assertTrue("pg_name1" in graph.used_process_group)
        pg_name_init_map.clear()
        pg_name_map.clear()

    def test_rename_cached_pgname(self):
        torch.distributed.init_process_group(
            backend='hccl', world_size=2, rank=0)
        unused_pg = torch.distributed.new_group()
        # 模拟convert调用
        self.assertEqual(get_group_name_and_record(
            "tag1", [0, 1], 2), 'pg_name1')
        self.assertTrue(default_pg.is_init_comm)
        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.HcomAllReduce(x, reduction="sum",
                                 group=default_pg.pg_name, fusion=0)
        rename_cached_pgname(graph._proto, {'pg_name1': ([0, 1], "tag1")})
        check_success = False
        for op in graph._proto.op:
            if op.type == "HcomAllReduce":
                self.assertTrue("group" in op.attr)
                from torchair.ge.attr import Str
                self.assertEqual(
                    Str.get(op.attr["group"]).value, encode_pg_tag_ranklist("tag1", [0, 1]))
                check_success = True
        self.assertTrue(check_success)
        self.assertFalse(unused_pg.is_init_comm)
        pg_name_init_map.clear()
        pg_name_map.clear()

    def test_build_cache_not_init_unuse_pg(self):
        torch.distributed.init_process_group(
            backend='hccl', world_size=2, rank=0)
        self.assertEqual(get_group_name_and_record(
            "tag1", [0, 1], 2), 'pg_name1')
        self.assertTrue(default_pg.is_init_comm)
        unused_pg = torch.distributed.new_group()
        with GeGraph() as ge_graph:
            x = ge.Data(index=0, shape=[1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.HcomAllReduce(x, reduction="sum",
                                 group=default_pg.pg_name, fusion=0)
        record_pg_to_graph(ge_graph)
        self.assertTrue("pg_name1" in ge_graph.used_process_group)
        self.assertEqual(ge_graph.used_process_group["pg_name1"], ([0, 1], "tag1"))
        rename_cached_pgname(ge_graph._proto, ge_graph.used_process_group)
        head = IndentedBuffer()
        head.writelines(['import torch', 'from torchair.ge._ge_graph import GeGraph',
                         f'serialized_graph = {ge_graph.SerializeToString()}'])
        head.writelines(['ge_graph = GeGraph(serialized_model_def=serialized_graph)'])
        code = codegen_refresh_cache_pgname(ge_graph.used_process_group)
        head.splice(code)
        exec(compile(head.getvalue(), '<string>', 'exec'))
        self.assertFalse(unused_pg.is_init_comm)
        pg_name_init_map.clear()
        pg_name_map.clear()

    def test_use_cache_not_init_unuse_pg(self):
        torch.distributed.init_process_group(
            backend='hccl', world_size=2, rank=0)
        unused_pg = torch.distributed.new_group()
        with GeGraph() as cache_graph:
            x = ge.Data(index=0, shape=[1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.HcomAllReduce(x, reduction="sum",
                                 group=encode_pg_tag_ranklist('tag1', [0, 1]), fusion=0)
        head = IndentedBuffer()
        head.writelines(['import torch', 'from torchair.ge._ge_graph import GeGraph',
                         f'serialized_graph = {cache_graph.SerializeToString()}'])
        head.writelines(['ge_graph = GeGraph(serialized_model_def=serialized_graph)'])
        code = codegen_refresh_cache_pgname({'pg_name1': ([0, 1], "tag1")})
        head.splice(code)
        exec(compile(head.getvalue(), '<string>', 'exec'))
        self.assertFalse(unused_pg.is_init_comm)
        self.assertTrue(default_pg.is_init_comm)
        self.assertEqual(len(PatchWorld.pg_map), 2)
        pg_name_init_map.clear()
        pg_name_map.clear()

    def test_use_ge_cache_no_pgname_init(self):
        torch.distributed.init_process_group(
            backend='hccl', world_size=2, rank=0)
        with GeGraph() as cache_graph:
            x = ge.Data(index=0, shape=[1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.HcomAllReduce(x, reduction="sum",
                                 group=encode_pg_tag_ranklist('tag1', [0, 1]), fusion=0)
        head = IndentedBuffer()
        head.writelines(['import torch', 'from torchair.ge._ge_graph import GeGraph',
                         f'serialized_graph = {cache_graph.SerializeToString()}'])
        head.writelines(['ge_graph = GeGraph(serialized_model_def=serialized_graph)'])
        code = codegen_refresh_cache_pgname({'pg_name1': ([0, 1], "tag1")})
        head.splice(code)
        exec(compile(head.getvalue(), '<string>', 'exec'))
        self.assertTrue(default_pg.is_init_comm)
        self.assertEqual(len(PatchWorld.pg_map), 1)
        self.assertEqual(default_pg.get_hccl_comm_name(0, init_comm=False), "pg_name1")
        pg_name_init_map.clear()
        pg_name_map.clear()

    def test_use_ge_cache_in_new_pgname_init(self):
        torch.distributed.init_process_group(
            backend='hccl', world_size=2, rank=0)
        default_pg.get_hccl_comm_name(0, init_comm=True)
        self.assertEqual(default_pg.get_hccl_comm_name(0, init_comm=False), "pg_name1")
        with GeGraph() as cache_graph:
            x = ge.Data(index=0, shape=[1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.HcomAllReduce(x, reduction="sum",
                                 group=encode_pg_tag_ranklist('tag1', [0, 1]), fusion=0)
        head = IndentedBuffer()
        head.writelines(['import torch', 'from torchair.ge._ge_graph import GeGraph',
                         f'serialized_graph = {cache_graph.SerializeToString()}'])
        head.writelines(['ge_graph = GeGraph(serialized_model_def=serialized_graph)'])
        code = codegen_refresh_cache_pgname({'pg_name2': ([0, 1], "tag1")})
        head.splice(code)
        exec(compile(head.getvalue(), '<string>', 'exec'))
        self.assertEqual(len(PatchWorld.pg_map), 1)
        pg_name_init_map.clear()
        pg_name_map.clear()

    def test_use_ge_cache_in_second_graph_used_old_pg(self):
        torch.distributed.init_process_group(
            backend='hccl', world_size=2, rank=0)
        default_pg.get_hccl_comm_name(0, init_comm=True)
        self.assertEqual(default_pg.get_hccl_comm_name(0, init_comm=False), "pg_name1")
        with GeGraph() as cache_graph:
            x = ge.Data(index=0, shape=[1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.HcomAllReduce(x, reduction="sum",
                                 group=encode_pg_tag_ranklist('tag1', [0, 1]), fusion=0)
        head = IndentedBuffer()
        head.writelines(['import torch', 'from torchair.ge._ge_graph import GeGraph',
                         f'serialized_graph = {cache_graph.SerializeToString()}'])
        head.writelines(['ge_graph = GeGraph(serialized_model_def=serialized_graph)'])
        code = codegen_refresh_cache_pgname({'pg_name2': ([0, 1], "tag1")})
        head.splice(code)
        exec(compile(head.getvalue(), '<string>', 'exec'))
        exec(compile(head.getvalue(), '<string>', 'exec'))
        pg_name_init_map.clear()
        pg_name_map.clear()
        self.assertEqual(len(PatchWorld.pg_map), 1)


if __name__ == '__main__':
    unittest.main()
