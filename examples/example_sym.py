import logging
import torch
import torch.nn as nn
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger

logger.setLevel(logging.DEBUG)

config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)


def test_autograd_static_sym():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            x = x + y
            z = torch.split(x[2:4, 2:4], y)
            return z

    model = Model()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    input0 = torch.randn(size=(4, 4), dtype=torch.float32, requires_grad=True).npu()
    res = model(input0, 1)
    assert res[0].size() == torch.Size([1, 2])
    loss_fn = nn.MSELoss()
    target_data = torch.randn((1, 2), requires_grad=True).to(torch.float32).npu()
    loss = loss_fn(res[0], target_data)
    loss.backward()


def test_autograd_dynamic_sym():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            x = x + y
            z = torch.split(x[2:4, 2:4], y)
            return z

    model = Model()
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    print(type(model))
    print((model))
    input0 = torch.randn(size=(4, 4), dtype=torch.float32, requires_grad=True).npu()
    res = model(input0, 1)
    assert res[0].size() == torch.Size([1, 2])
    loss_fn = nn.MSELoss()
    target_data = torch.randn((1, 2), requires_grad=True).to(torch.float32).npu()
    loss = loss_fn(res[0], target_data)
    loss.backward()


def mp():
    # == == == == == == == == =  case1 sym动态返回 == == == == == == == == ==
    test_autograd_dynamic_sym()
    print("==================case 1 pass =============================", flush=True)
    # # == == == == == == == == =  case1 sym场景静态返回 == == == == == == == == ==
    test_autograd_static_sym()
    print("==================case 2 pass =============================", flush=True)


if __name__ == '__main__':
    mp()
