import torch
import torch_npu
import torchair
from torchair.ge_concrete_graph import ge_graph as grp

dtype_torch = [torch.float32, torch.int32, torch.bool, torch.float16, 
               torch.int8, torch.uint8, torch.int16, torch.int64,
               torch.float64, torch.bfloat16, torch.complex64, torch.complex128, 
               torch.qint8, torch.quint8, torch.qint32]
dtype_ge = [grp.DataType.DT_FLOAT16, grp.DataType.DT_FLOAT, grp.DataType.DT_DOUBLE, 
            grp.DataType.DT_INT8, grp.DataType.DT_UINT8, grp.DataType.DT_INT32, 
            grp.DataType.DT_UINT32, grp.DataType.DT_INT64, grp.DataType.DT_BOOL, 
            grp.DataType.DT_BF16, grp.DataType.DT_INT16, grp.DataType.DT_COMPLEX64, 
            grp.DataType.DT_COMPLEX128, grp.DataType.DT_QINT8, grp.DataType.DT_QUINT8, 
            grp.DataType.DT_QINT32]

for i in dtype_torch:
    grp.torch_type_to_ge_type(i)

for i in dtype_ge:
    grp._ge_dtype_to_ge_proto_dtype(i)

print("success run all dtype")