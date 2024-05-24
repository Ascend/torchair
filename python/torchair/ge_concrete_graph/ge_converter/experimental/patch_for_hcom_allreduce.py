from torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce import backup_custom_all_reduce
from torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce import get_npu_all_reduce
from torchair.core.utils import logger

try :
    from deepspeed import comm as dist
except:
    logger.info(f'env not import deepspeed, only patch pytorch dist api')
else:
    backup_custom_all_reduce(dist.all_reduce)
    dist.all_reduce = get_npu_all_reduce()
    # Adapt deepspeed version later than v0.10.0,
    # inference_all_reduce and all_reduce have save torch backend implementation.
    dist.inference_all_reduce = get_npu_all_reduce()