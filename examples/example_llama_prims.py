from typing import Any, Dict, List, Tuple, Union
from torch._functorch.aot_autograd import aot_module_simplified
import torch

import logging
from torchair.core.utils import logger
from torchair.configs.compiler_config import CompilerConfig
import torchair as tng
from torch._decomp import get_decompositions
aten = torch.ops.aten

logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.aoe_config.aoe_mode = "1"
config.debug.graph_dump.type = "pbtxt"

custom_decompositions=get_decompositions([
    aten.add,
    aten.lt,
    aten.matmul,
    aten.mul,
    aten.cumsum,
    aten.maximum,
    aten.minimum,
    aten.le,
    aten.addmm,
])
npu_backend = tng.get_npu_backend(compiler_config=config, custom_decompositions=custom_decompositions)

target_dtype = torch.float16


class LlamaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.var = 9.9

    def forward(self, x, y):
        add1 = x + (self.var)
        add2 = add1 + x
        mul1 = y.view([-1]) * 1.0
        mul2 = mul1 * add2
        view1 = mul2.view([-1, 128])
        arange1 = torch.arange(view1.size(-1))
        unsquese1 = torch.unsqueeze(arange1, 0)
        clone1 = unsquese1.clone()
        transpose1 = torch.transpose(clone1, 0, 1)
        div1 = view1 / 0.1
        mm1 = torch.matmul(div1, transpose1.to(torch.int).to(target_dtype))
        expand1 = mm1.expand([64, 128])
        full1 = torch.full((mm1.size(0), mm1.size(0)), 1.0)
        cat1 = torch.cat([full1, -full1], -1)
        # expand1.masked_fill_(cat1.to(torch.bool), 1)
        expand1 = expand1 + cat1
        pow1 = expand1.pow(2)
        repeat = pow1.repeat([2, 1, 1])
        unsafe_view1 = repeat.reshape(
            repeat.size(0), repeat.size(2), repeat.size(1))
        bmm1 = torch.bmm(repeat, unsafe_view1)
        indices = bmm1[1:2, :32, :-1].to(torch.bool).to(torch.int64)
        gather1 = torch.gather(bmm1, 1, indices)
        softmax1 = torch.nn.functional.softmax(
            gather1, dim=-1, dtype=target_dtype)
        rsqrt1 = torch.rsqrt(softmax1)
        rsub = 1.0 - rsqrt1
        silu = torch.nn.functional.silu(rsub)
        eb1 = torch.embedding(y, silu.view(
            [32, -1]).to(torch.long), -1, False, False)
        lt = silu[:1, :1, :1] < eb1[:1, :1, :1]
        return lt


model = LlamaModel()
model = torch.compile(model, backend=npu_backend, dynamic=True)
ins1 = torch.randn([8192]).to(target_dtype)
ins2 = torch.randn([64, 128]).to(target_dtype)
model(ins1, ins2)
ins3 = torch.randn([128, 64]).to(target_dtype)
model(ins1, ins3)
