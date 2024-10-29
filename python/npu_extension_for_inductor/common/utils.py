import re
from enum import Enum
from collections import namedtuple
import torch

TypeSet = namedtuple("TypeSet", ['torch', 'acl'])


class AclDataType(Enum):
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
    TypeSet(torch.float32, AclDataType.ACL_FLOAT.value),
    TypeSet(torch.int32, AclDataType.ACL_INT32.value),
    TypeSet(torch.bool, AclDataType.ACL_BOOL.value),
    TypeSet(torch.float16, AclDataType.ACL_FLOAT16.value),
    TypeSet(torch.int8, AclDataType.ACL_INT8.value),
    TypeSet(torch.uint8, AclDataType.ACL_UINT8.value),
    TypeSet(torch.int16, AclDataType.ACL_INT16.value),
    TypeSet(torch.int64, AclDataType.ACL_INT64.value),
    TypeSet(torch.float64, AclDataType.ACL_DOUBLE.value),
    TypeSet(torch.bfloat16, AclDataType.ACL_BF16.value),
]


def _torch_type_to_type(torch_type, dst):
    ret = None
    for type_set in _ALL_TYPES:
        if type_set.torch == torch_type:
            ret = getattr(type_set, dst)

    if ret is None:
        raise RuntimeError(f"Unsupported torch type {torch_type} by {dst}")
    return ret


class StrRep:
    def __init__(self, value, str_value=None):
        self.value = value
        self.str_value = str_value if str_value else value

    def __str__(self):
        return self.str_value

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(self.value)

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, value: object) -> bool:
        return self.value == value


class TypeUtils:
    @classmethod
    def torch_to_acl(cls, dtype):
        return _torch_type_to_type(dtype, "acl")

    @classmethod
    def torch_to_asc(cls, dtype):
        if dtype is None:
            return dtype
        type_str = str(dtype).split('torch.')[-1]
        return StrRep(f"ascir.dtypes.{type_str}", type_str)


def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
