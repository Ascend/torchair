import unittest
import time
import os
import shutil
import logging
import torch
import torch.distributed._functional_collectives as funcol
import torchair
import torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce
from torchair.core.utils import logger

os.environ['TNG_LOG_LEVEL'] = '0'
logger.setLevel(logging.DEBUG)


def get_dumped_file_list(dir_path, file_extension='.pbtxt'):
    return [i for i in os.listdir(dir_path) if i.startswith('dynamo') and i.endswith(f'{file_extension}')]


class TorchairSt(unittest.TestCase):
    def setUp(self) -> None:
        self.clean_env()
        return super().setUp()

    def tearDown(self) -> None:
        self.clean_env()
        return super().tearDown()

    def clean_env(self):
        for export_path in ["export_file", "false_export_path2", "true_export_path2", \
                            "true_export_path3", "test_export_file_path", "test_export_file_path2"]:
            if os.path.exists(export_path):
                shutil.rmtree(export_path)

    def test_export(self):
        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + y
                return x

        model = Model()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)

        export_path1 = "test_export_file_path"
        if os.path.exists(export_path1):
            shutil.rmtree(export_path1)
        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=False)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()

        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"Shape\"") == 0

    def test_export_with_sym(self):
        def get_inputnum_in_node(strgraph, opname):
            start_str = opname
            end_str = "attr {"
            start_index = strgraph.find(start_str)
            sub_str = strgraph[start_index: len(strgraph) - 1]
            end_index = sub_str.find(end_str)
            result = sub_str[0: end_index]
            return result.count("input: ")

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + y
                z = torch.cat((x, y), 0)
                return z.size()[1], x

        model = Model()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)

        export_path1 = "test_export_file_path"
        if os.path.exists(export_path1):
            shutil.rmtree(export_path1)

        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=True)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()

        assert src.count("op: \"Shape\"") == 1
        assert src.count("op: \"Data\"") == 2
        assert get_inputnum_in_node(src, "op: \"NetOutput\"")

    def test_export_with_allreduce(self):
        def get_sub_path_dynamo_pbtxt(export_path, rankid):
            return export_path + "/rank_" + str(rankid) + "/dynamo.pbtxt"

        def get_model_relation_config(export_path):
            return export_path + "/model_relation_config.json"

        def get_numa_config(export_path):
            return export_path + "/numa_config.json"

        def mp():
            world_size = 2
            torch.multiprocessing.spawn(example, args=(world_size,), nprocs=world_size, join=True)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        mp()

        file_name = get_sub_path_dynamo_pbtxt("export_file", 0)
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("op: \"Const\"") == 2
        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"HcomAllReduce\"") == 1
        assert src.count("key: \"ranklist\"") == 1

        file_name = get_sub_path_dynamo_pbtxt("false_export_path2", 0)
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("op: \"HcomAllReduce\"") == 4  # 多group场景

        file_name = get_sub_path_dynamo_pbtxt("true_export_path2", 0)
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count(" dim: -1") == 3  # 动态图存在-1

        file_name = get_model_relation_config("true_export_path2")
        # mutil group case, can not create atc config file
        assert os.path.exists(file_name) == False
        file_name = get_numa_config("true_export_path2")
        assert os.path.exists(file_name) == False

        file_name = get_sub_path_dynamo_pbtxt("true_export_path3", 0)
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("HcomReduceScatter") == 3  # dist reduce_scatter_tensor入图

    def test_export_weight_externalized(self):
        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.randn([1024, 1024, 1024], dtype=torch.float16))  # 2G weight

            def forward(self, x, y):
                x = x + y
                w = self.p1 * 2
                return x, w

        model = Model()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)

        export_path1 = "test_export_file_path"

        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=False)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()

        assert src.count("op: \"FileConstant\"") == 1
        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"Shape\"") == 0

    def test_export_with_atc_config_generated(self):
        def get_sub_path_dynamo_pbtxt(export_path, rankid):
            return export_path + "/rank_" + str(rankid) + "/dynamo.pbtxt"

        def get_model_relation_config(export_path):
            return export_path + "/model_relation_config.json"

        def get_numa_config(export_path):
            return export_path + "/numa_config.json"

        def mp():
            world_size = 2
            torch.multiprocessing.spawn(example_atc_config_generated, args=(world_size,), nprocs=world_size, join=True)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        mp()

        file_name = get_sub_path_dynamo_pbtxt("export_file", 0)
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("op: \"Const\"") == 2
        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"HcomAllReduce\"") == 1
        assert src.count("key: \"ranklist\"") == 1

        file_name = get_model_relation_config("export_file")
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("\"submodel_name\": \"export_rank0.air\"") == 2
        assert src.count("\"group_rank_list\": \"[0, 1]\"") == 1
        assert src.count("model_instance_id") == 4
        assert src.count("0:0:0") == 1
        assert src.count("0:0:1") == 1

        file_name = get_numa_config("export_file")
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("\"item_id\": 0") == 1
        assert src.count("\"item_id\": 1") == 1

    def test_export_bf16(self):
        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                size = 10
                self.p1 = torch.nn.Parameter(torch.randn([size], dtype=torch.bfloat16))

            def forward(self, x, y):
                x = x + y + torch.ones([2, 4], dtype=torch.float16)
                w = self.p1 * 2
                return x, w

        model = Model()
        x = torch.randn([2, 4], dtype=torch.bfloat16)
        y = torch.randn([2, 4], dtype=torch.bfloat16)

        export_path1 = "test_export_file_path"

        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=False)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()

        assert src.count("op: \"Const\"") == 4
        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"Shape\"") == 0
        assert src.count("dtype: DT_BF16") == 13
        assert src.count("  dim: 10") == 3

    def test_export_enable_record_nn_module_stack(self):
        class Model2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x + y
                return torch.chunk(z, 2, dim=0)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodel = Model2()

            def forward(self, x, y):
                x = x * 2
                return self.submodel(x, y)

        model = Model()
        x = torch.randn([2, 4], dtype=torch.bfloat16)
        y = torch.randn([2, 4], dtype=torch.float16)
        from torchair.configs.compiler_config import CompilerConfig
        export_path1 = "test_export_file_path"
        config = CompilerConfig()
        config.export.experimental.enable_record_nn_module_stack = True
        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=False, config=config)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("\"nn_module_stack\"") == 6  # 插入了2个cast,有两个输出
        assert src.count("Model2") == 6

    def test_export_with_sym_pack(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z, p, m, n):
                b = p.view([x]) + z.view([x]) + z.view([x])
                c = m.view([x, x]).sum()
                a = torch.stack([n, n, n, n])
                d = (m.view([4, y, y]) + a).sum()
                return b + c - d

        model = Model()
        z = torch.randn([3, 2])
        p = torch.randn([2, 3])
        m = torch.randn([36])
        n = torch.randn([3, 3])

        export_path1 = "test_export_file_path"
        if os.path.exists(export_path1):
            shutil.rmtree(export_path1)
        torchair.dynamo_export(6, 3, z, p, m, n, model=model, export_path=export_path1, dynamic=True)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()

        assert src.count("op: \"Data\"") == 6
        assert src.count("op: \"Pack\"") == 2

    def test_export_with_ref_data(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x * x
                return y.add_(a)

        model = Model()
        export_path = "test_export_file_path"
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        x = torch.randn([2, 4], dtype=torch.float16)
        y = torch.randn([2, 4], dtype=torch.float16)
        torchair.dynamo_export(x, y, model=model, export_path=export_path, dynamic=False)

    def test_one_graph_with_same_ref_data(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x.mul_(2)
                return x, y.add_(1)

        model = Model()
        export_path = "test_export_file_path"
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        x = torch.randn([2, 4], dtype=torch.float16)
        torchair.dynamo_export(x, x, model=model, export_path=export_path, dynamic=False)

        dumped_file_list = get_dumped_file_list(export_path)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src1 = f.read()

        shape_str = "_".join(str(sh) for sh in x.shape)
        stride_str = "_".join(str(std) for std in x.stride())
        offset_str = str(x.storage_offset())
        new_refdata_name = "RefData__" + shape_str + "__" + stride_str + "__" + offset_str + "__" + str(
            id(x))

        name_str = f'name: \"{new_refdata_name}\"'
        assert src1.count(name_str) == 1
        op_str = f'op: "RefData"'
        assert src1.count(op_str) == 1



    def test_two_graph_with_same_ref_data(self):
        class Model1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x * x
                return y.add_(a)
        class Model2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = torch.mm(x, x)
                b = a + x
                return y.mul_(b)

        model = Model1()
        export_path1 = "test_export_file_path"
        export_path2 = "test_export_file_path2"

        if os.path.exists(export_path1):
            shutil.rmtree(export_path1)
        input_tensor1 = torch.randn([2, 4], dtype=torch.float16)
        input_tensor2 = torch.randn([2, 4], dtype=torch.float16)
        print(id(input_tensor1))
        print(id(input_tensor2))

        torchair.dynamo_export(input_tensor1, input_tensor2, model=model, export_path=export_path1, dynamic=False)

        torchair.dynamo_export(input_tensor1, input_tensor2, model=model, export_path=export_path2, dynamic=False)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src1 = f.read()

        shape_str = "_".join(str(x) for x in input_tensor2.shape)
        stride_str = "_".join(str(x) for x in input_tensor2.stride())
        offset_str = str(input_tensor2.storage_offset())
        new_refdata_name = "RefData__" + shape_str + "__" + stride_str + "__" + offset_str + "__" + str(
            id(input_tensor2))

        sub_str = f'name: \"{new_refdata_name}\"'
        assert src1.count(sub_str) == 1

        dumped_file_list = get_dumped_file_list(export_path2)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path2, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path2, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src2 = f.read()
        assert src2.count(sub_str) == 1


class AllReduceSingeGroup(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor([[1.1, 1.1], [1.1, 1.1]]))
        self.p2 = torch.nn.Parameter(torch.tensor([[2.2, 2.2], [3.3, 3.3]]))

    def forward(self, x, y):
        x = x + y + self.p + self.p2
        torch.distributed.all_reduce(x)
        return x


class AllReduceMultiGroup(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test1')
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test2')
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test3')
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test1')  # 重复的group case
        return x


class DistReduceScatterTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, output):
        from torch.distributed.distributed_c10d import _world
        # 必须要带group参数
        torch.distributed.reduce_scatter_tensor(output, x, group=_world.default_pg)
        return x


class FuncolReduceScatterTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        from torch.distributed.distributed_c10d import _world
        out = funcol.reduce_scatter_tensor(x, "sum", scatter_dim=-1, group=_world.default_pg)
        return out


def example(rank, world_size):
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    x = torch.ones([2, 2], dtype=torch.int32)
    y = torch.ones([2, 2], dtype=torch.int32)
    mod = AllReduceSingeGroup()
    torchair.dynamo_export(x, y, model=mod)

    mod2 = AllReduceMultiGroup()
    xx2 = torch.ones([3], dtype=torch.int32)
    torchair.dynamo_export(xx2, model=mod2, dynamic=False, export_path="false_export_path2")
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.export.experimental.auto_atc_config_generated = True
    torchair.dynamo_export(xx2, model=mod2, dynamic=True, export_path="true_export_path2",
                           config=config)

    mod3 = DistReduceScatterTensor()
    xx3 = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    output = torch.empty([2], dtype=torch.int32)
    torchair.dynamo_export(xx3, output, model=mod3, dynamic=True, export_path="true_export_path3")


def example_atc_config_generated(rank, world_size):
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    x = torch.ones([2, 2], dtype=torch.int32)
    y = torch.ones([2, 2], dtype=torch.int32)
    mod = AllReduceSingeGroup()
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.export.experimental.auto_atc_config_generated = True
    torchair.dynamo_export(x, y, model=mod, config=config)


if __name__ == '__main__':
    unittest.main()
