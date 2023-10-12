import unittest
import time
import os
import shutil
import logging
import torch
import torchair
import torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce
from torchair.core.utils import logger

os.environ['TNG_LOG_LEVEL'] = '0'
logger.setLevel(logging.DEBUG)


class TorchairSt(unittest.TestCase):
    def test_export(self):
        def get_dumped_file_list(dir_path, file_extension='.pbtxt'):
            return [i for i in os.listdir(dir_path) if i.startswith('dynamo') and i.endswith(f'{file_extension}')]

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + y
                return x


        model = Model()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)

        export_path1 = "./test_export_file_False"
        if os.path.exists(export_path1):
            shutil.rmtree(export_path1)
        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=False)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r')as f:
            src = f.read()

        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"Shape\"") == 0

    def test_export_with_sym(self):
        def get_dumped_file_list(dir_path, file_extension='.pbtxt'):
            return [i for i in os.listdir(dir_path) if i.startswith('dynamo') and i.endswith(f'{file_extension}')]

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

        export_path1 = "./test_export_file_True"
        if os.path.exists(export_path1):
            shutil.rmtree(export_path1)

        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=True)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r')as f:
            src = f.read()

        assert src.count("op: \"Shape\"") == 1
        assert src.count("op: \"Data\"") == 2
        assert get_inputnum_in_node(src, "op: \"NetOutput\"")

    def test_export_with_allreduce(self):
        def get_dumped_file_list(dir_path, file_extension='.pbtxt'):
            return [i for i in os.listdir(dir_path) if i.startswith('dynamo') and i.endswith(f'{file_extension}')]

        def mp():
            world_size = 2
            torch.multiprocessing.spawn(example, args=(world_size, ), nprocs=world_size, join=True)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        export_path1 = "./export_file_rank_0"
        if os.path.exists(export_path1):
            shutil.rmtree(export_path1)
        export_path2 = "./export_file_rank_1"
        if os.path.exists(export_path2):
            shutil.rmtree(export_path2)

        mp()

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r')as f:
            src = f.read()

        assert src.count("op: \"FileConstant\"") == 2
        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"HcomAllReduce\"") == 1
        assert src.count("key: \"ranklist\"") == 1


class HcomModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor([[1.1, 1.1], [1.1, 1.1]]))
        self.p2 = torch.nn.Parameter(torch.tensor([[2.2, 2.2], [3.3, 3.3]]))

    def forward(self, x, y):
        x = x + y + self.p + self.p2
        torch.distributed.all_reduce(x)
        return x


def example(rank, world_size):
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    x = torch.ones([2, 2], dtype=torch.int32)
    y = torch.ones([2, 2], dtype=torch.int32)
    mod = HcomModel()
    torchair.dynamo_export(x, y, model=mod)
    torchair.dynamo_export(x, y, model=mod, dynamic=True)

if __name__ == '__main__':
    unittest.main()
