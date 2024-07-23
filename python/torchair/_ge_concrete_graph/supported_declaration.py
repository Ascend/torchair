import torch


class _TypedTensor:
    def __init__(self, *dims, dtype, value=None, value_range=None):
        self.dims = dims
        self.dtype = dtype
        self.value = value
        if value_range is not None and len(value_range) != 2:
            raise AssertionError
        self.value_range = value_range

    def __str__(self) -> str:
        if self.value is not None:
            return f"tensor({self.value}, dtype={self.dtype})"
        if self.value_range is not None:
            return f"tensor({self.dims}, dtype={self.dtype}, value_range={self.value_range})"
        return f"Tensor({self.dims}, {self.dtype})"

    def __repr__(self) -> str:
        if self.value is not None:
            return f"tensor({self.value}, dtype={self.dtype})"
        if self.value_range is not None:
            return f"tensor({self.dims}, dtype={self.dtype}, value_range={self.value_range})"
        return f"Tensor({self.dims}, {self.dtype})"


class F32(_TypedTensor):
    def __init__(self, *dims, value_range=None):
        super().__init__(*dims, dtype=torch.float32, value_range=value_range)


class F16(_TypedTensor):
    def __init__(self, *dims, value_range=None):
        super().__init__(*dims, dtype=torch.float16, value_range=value_range)


class BF16(_TypedTensor):
    def __init__(self, *dims):
        super().__init__(*dims, dtype=torch.bfloat16)


class F64(_TypedTensor):
    def __init__(self, *dims, value_range=None):
        super().__init__(*dims, dtype=torch.float64, value_range=value_range)


class I32(_TypedTensor):
    def __init__(self, *dims, value_range=None):
        super().__init__(*dims, dtype=torch.int32, value_range=value_range)


class I16(_TypedTensor):
    def __init__(self, *dims, value_range=None):
        super().__init__(*dims, dtype=torch.int16, value_range=value_range)


class I64(_TypedTensor):
    def __init__(self, *dims, value_range=None):
        super().__init__(*dims, dtype=torch.int64, value_range=value_range)


class I8(_TypedTensor):
    def __init__(self, *dims, value_range=None):
        super().__init__(*dims, dtype=torch.int8, value_range=value_range)


class U8(_TypedTensor):
    def __init__(self, *dims, value_range=None):
        super().__init__(*dims, dtype=torch.uint8, value_range=value_range)


class C64(_TypedTensor):
    def __init__(self, *dims, value_range=None):
        super().__init__(*dims, dtype=torch.complex64, value_range=value_range)


class BOOL(_TypedTensor):
    def __init__(self, *dims):
        super().__init__(*dims, dtype=torch.bool)


class T(_TypedTensor):
    def __init__(self, value, *, dtype):
        super().__init__(dtype=dtype, value=value)


class Support:
    def __init__(self, *args, **kwargs) -> None:
        def is_tensor_assert(arg):
            if isinstance(arg, torch.Tensor):
                raise AssertionError

        for arg in args:
            if isinstance(arg, (list, tuple)):
                for v in arg:
                    is_tensor_assert(v)
            else:
                is_tensor_assert(arg)
        for k, v in kwargs.items():
            is_tensor_assert(v)
        self.args = args
        self.kwargs = kwargs
        self.title = ""

    def __str__(self) -> str:
        def _format(args, kwargs):
            all_args = [f"args[{i}] = {v}" for i, v in enumerate(args)]
            all_args.extend(
                [f"kwargs['{k}'] = {v}" for k, v in kwargs.items()])
            return "\n  " + "\n  ".join(all_args)
        return _format(self.args, self.kwargs)
