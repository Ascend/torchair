from enum import Enum

import torch


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


def torch_type_to_acl_type(dtype):
    if dtype == torch.float32:
        return aclDataType.ACL_FLOAT.value
    if dtype == torch.int32:
        return aclDataType.ACL_INT32.value
    if dtype == torch.bool:
        return aclDataType.ACL_BOOL.value
    if dtype == torch.float16:
        return aclDataType.ACL_FLOAT16.value
    if dtype == torch.int8:
        return aclDataType.ACL_INT8.value
    if dtype == torch.uint8:
        return aclDataType.ACL_UINT8.value
    if dtype == torch.int16:
        return aclDataType.ACL_INT16.value
    if dtype == torch.int64:
        return aclDataType.ACL_INT64.value
    if dtype == torch.float64:
        return aclDataType.ACL_DOUBLE.value
    if dtype == torch.bfloat16:
        return aclDataType.ACL_BF16.value

    raise RuntimeError(f"Unsupported torch type {dtype} by acl")
