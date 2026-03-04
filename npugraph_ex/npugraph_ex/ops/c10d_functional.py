from typing import List, Optional

import torch
import torch.distributed._functional_collectives

from npugraph_ex.ops._hcom_alltoall import _all_to_all_single


if torch.__version__ >= '2.3.1':
    def decomp_c10d_functional_all_to_all_single(
        input_tensor: torch.Tensor,
        output_split_sizes: Optional[List[int]],
        input_split_sizes: Optional[List[int]],
        group_name: str,
    ):
        group = torch.distributed.distributed_c10d._resolve_process_group(group_name)
        rank_list = torch.distributed.get_process_group_ranks(group)
        tag = torch.distributed.distributed_c10d._get_group_tag(group)
        return _all_to_all_single(input_tensor,
                                  output_split_sizes,
                                  input_split_sizes,
                                  tag,
                                  rank_list,
                                  len(rank_list))
