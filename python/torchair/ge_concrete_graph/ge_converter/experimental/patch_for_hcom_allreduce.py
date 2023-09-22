from torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce import backup_custom_all_reduce
from torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce import get_npu_all_reduce
from deepspeed import comm as dist

backup_custom_all_reduce(dist.all_reduce)
dist.all_reduce = get_npu_all_reduce()
