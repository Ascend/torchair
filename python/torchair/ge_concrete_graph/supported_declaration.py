import torch


class _TypedTensor:
    def __init__(self, *dims, dtype, value=None):
        self.dims = dims
        self.dtype = dtype
        self.value = value

    def __str__(self) -> str:
        if self.value is not None:
            return f"tensor({self.value}, dtype={self.dtype})"
        return f"Tensor({self.dims}, {self.dtype})"

    def __repr__(self) -> str:
        if self.value is not None:
            return f"tensor({self.value}, dtype={self.dtype})"
        return f"Tensor({self.dims}, {self.dtype})"


class F32(_TypedTensor):
    def __init__(self, *dims):
        super().__init__(*dims, dtype=torch.float32)


class F16(_TypedTensor):
    def __init__(self, *dims):
        super().__init__(*dims, dtype=torch.float16)


class F64(_TypedTensor):
    def __init__(self, *dims):
        super().__init__(*dims, dtype=torch.float64)


class I32(_TypedTensor):
    def __init__(self, *dims):
        super().__init__(*dims, dtype=torch.int32)


class I16(_TypedTensor):
    def __init__(self, *dims):
        super().__init__(*dims, dtype=torch.int16)


class I64(_TypedTensor):
    def __init__(self, *dims):
        super().__init__(*dims, dtype=torch.int64)


class I8(_TypedTensor):
    def __init__(self, *dims):
        super().__init__(*dims, dtype=torch.int8)


class U8(_TypedTensor):
    def __init__(self, *dims):
        super().__init__(*dims, dtype=torch.uint8)


class BOOL(_TypedTensor):
    def __init__(self, *dims):
        super().__init__(*dims, dtype=torch.bool)


class T(_TypedTensor):
    def __init__(self, value, *, dtype):
        super().__init__(dtype=dtype, value=value)


class Support:
    def __init__(self, *args, **kwargs) -> None:
        for arg in args:
            if isinstance(arg, (list, tuple)):
                for v in arg:
                    assert not isinstance(v, torch.Tensor)
            else:
                assert not isinstance(arg, torch.Tensor)
        for k, v in kwargs.items():
            assert not isinstance(v, torch.Tensor)
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
