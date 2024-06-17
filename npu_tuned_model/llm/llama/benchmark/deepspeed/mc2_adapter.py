# coding=utf-8
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch import nn
from deepspeed import comm as dist
import torch_npu


class LinearAllreduce(nn.Module):
    def __init__(self, weight, bias=None, mp_group=None):
        super(LinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group
        if self.mp_group is not None:
            rank = torch.distributed.get_rank(self.mp_group)
            global_rank = torch.distributed.get_global_rank(self.mp_group, rank)
            self.hcomm_info = self.mp_group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)

    def forward(self, input):
        bs, seq, hidden_size = input.shape
        if seq == 1:
            output = torch.matmul(input, self.weight.transpose(-1, -2))
            if self.mp_group is not None:
                dist.all_reduce(output, group=self.mp_group)
            if self.bias is not None:
                output += self.bias
        else:
            output = torch_npu.npu_mm_all_reduce_base(input, self.weight.transpose(-1, -2).contiguous(),
                                                      self.hcomm_info)

        return output
