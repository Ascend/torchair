from enum import Enum
from collections import namedtuple
import torch

TypeSet = namedtuple("TypeSet", ['torch', 'acl'])


class aclDataType(Enum):
    ACL_DT_UNDEFINED = -1
    ACL_FLOAT = 0
    ACL_FLOAT16 = 1
    ACL_INT8 = 2
    ACL_INT32 = 3
    ACL_UINT8 = 4
    ACL_INT16 = 6
    ACL_UINT16 = 7
    ACL_UINT32 = 8
    ACL_INT64 = 9
    ACL_UINT64 = 10
    ACL_DOUBLE = 11
    ACL_BOOL = 12
    ACL_STRING = 13
    ACL_COMPLEX64 = 16
    ACL_COMPLEX128 = 17
    ACL_BF16 = 27
    ACL_COMPLEX32 = 33


_ALL_TYPES = [
    TypeSet(torch.float32, aclDataType.ACL_FLOAT.value),
    TypeSet(torch.int32, aclDataType.ACL_INT32.value),
    TypeSet(torch.bool, aclDataType.ACL_BOOL.value),
    TypeSet(torch.float16, aclDataType.ACL_FLOAT16.value),
    TypeSet(torch.int8, aclDataType.ACL_INT8.value),
    TypeSet(torch.uint8, aclDataType.ACL_UINT8.value),
    TypeSet(torch.int16, aclDataType.ACL_INT16.value),
    TypeSet(torch.int64, aclDataType.ACL_INT64.value),
    TypeSet(torch.float64, aclDataType.ACL_DOUBLE.value),
    TypeSet(torch.bfloat16, aclDataType.ACL_BF16.value),
]


def _torch_type_to_type(torch_type, dst):
    ret = None
    for type_set in _ALL_TYPES:
        if type_set.torch == torch_type:
            ret = getattr(type_set, dst)

    if ret is None:
        raise RuntimeError(f"Unsupported torch type {torch_type} by {dst}")
    return ret


class TypeUtils:
    @classmethod
    def torch_to_acl(cls, dtype):
        return _torch_type_to_type(dtype, "acl")

    @classmethod
    def torch_to_asc(cls, dtype):
        return f"ascir.dtypes.{str(dtype).split('torch.')[1]}"
