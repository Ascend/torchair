import os
import sys
import torch
from torch import nn
import torch_npu


class NpuA8W8Linear(torch.nn.Module):
    __constants__ = ["in_feature", "out_feature"]
    in_feature: int
    out_feature: int

    def __init__(self, in_feature: int, out_feature: int, device="npu",
                output_dtype=torch.float16) -> None:
        super().__init__()
        self.output_dtype = output_dtype
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        ## A8W8 weight Perchannel symmetry, x pertensor asymmetry
        self.register_buffer("bias", torch.ones(self.out_feature, dtype=torch.int32, device="npu"))
        self.register_buffer("weight", torch.ones((self.out_feature, self.in_feature), dtype=torch.int8, device="npu"))
        self.register_buffer("deq_scale", torch.ones(self.out_feature, dtype=torch.int64, device="npu"))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_feature, )
        x = x.view(-1, x.size(-1))

        y = torch_npu.npu_quant_matmul(x,
                                       self.weight.transpose(0, 1),
                                       self.deq_scale,
                                       offset=None,
                                       bias=self.bias,
                                       output_dtype=self.output_dtype)

        return y.view(size_out)