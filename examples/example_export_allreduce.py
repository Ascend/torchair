import os
import logging
from typing import Any, Dict, List, Tuple, Union
import torch.multiprocessing as mp
import torch
import torch.distributed as dist

import torchair
from torchair.core.utils import logger
import torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce

logger.setLevel(logging.DEBUG)


class MyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor([[1.1, 1.1], [1.1, 1.1]]))
        self.p2 = torch.nn.Parameter(torch.tensor([[2.2, 2.2], [3.3, 3.3]]))

    def forward(self, x, y):
        x = x + y + self.p + self.p2
        torch.distributed.all_reduce(x)
        return x


def example(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    a = torch.ones([2, 2], dtype=torch.int32)
    b = torch.ones([2, 2], dtype=torch.int32)
    mod = MyModel()
    torchair.dynamo_export(a, b, model=mod)
    torchair.dynamo_export(a, b, model=mod, dynamic=True)


def main():
    world_size = 2
    mp.spawn(example, args=(world_size, ), nprocs=world_size, join=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    main()
