from typing import List, Dict
import torch
from torchair.core import _torchair


def _check_inner(arg_name, arg_value, inner_class):
    for val in arg_value:
        if not isinstance(val, inner_class):
            raise TypeError(f"{arg_name} inner type only support {[inner_class]}, but got {format(type(val))}.")


def _check_isinstance(arg_name, arg_value, classes, inner_class=None):
    check = False
    for clazz in classes:
        if isinstance(arg_value, clazz):
            if inner_class:
                _check_inner(arg_name, arg_value, inner_class)
            check = True
            break
    if not check:
        raise TypeError(
            f"{arg_name} only support {[clazz.__name__ for clazz in classes]}, but got {format(type(arg_value))}.")
    return arg_value


_TORCH_DTYPE_TO_GE_TYPE: Dict[torch.dtype, int] = {
    torch.bool: _torchair.TorchDataType.BOOL,
    torch.uint8: _torchair.TorchDataType.UINT8,
    torch.int8: _torchair.TorchDataType.INT8,
    torch.int16: _torchair.TorchDataType.INT16,
    torch.int32: _torchair.TorchDataType.INT32,
    torch.int64: _torchair.TorchDataType.INT64,
    torch.bfloat16: _torchair.TorchDataType.BF16,
    torch.float16: _torchair.TorchDataType.FLOAT16,
    torch.float: _torchair.TorchDataType.FLOAT32,
    torch.float32: _torchair.TorchDataType.FLOAT32,
    torch.float64: _torchair.TorchDataType.FLOAT64,
    torch.complex64: _torchair.TorchDataType.COMPLEX64,
    torch.complex128: _torchair.TorchDataType.COMPLEX128,
}


def _data_type_from_torch(dtype: torch.dtype) -> int:
    _check_isinstance('dtype', dtype, [torch.dtype])
    if dtype not in _TORCH_DTYPE_TO_GE_TYPE:
        raise ValueError(f'unsupported dtype: {dtype}, supported = {list(_TORCH_DTYPE_TO_GE_TYPE.keys())}')
    dtype_val = _TORCH_DTYPE_TO_GE_TYPE[dtype]
    return dtype_val


def _create_npu_tensors(shape: List[int], dtype: torch.dtype, addresses: List[int]) -> List[torch.Tensor]:
    _check_isinstance('shape', shape, [list, tuple], int)
    _check_isinstance('addresses', addresses, [list, tuple], int)
    data_type = _data_type_from_torch(dtype)
    ret, tensors = _torchair.as_torch_tensors(shape, data_type, addresses)
    if ret != 0:
        raise RuntimeError(f'llm_link_torch_wrapper.as_torch_tensors() returned {ret}')
    return tensors
