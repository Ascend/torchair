import torch.library
import torch

torch.utils.rename_privateuse1_backend("npu")


npulib = torch.library.Library("npu", "FRAGMENT")
npulib.define("npu_scatter_nd_update_(Tensor(a!) x, Tensor indices, Tensor updates) -> Tensor(a!)")
npulib.define("_npu_dtype_cast(Tensor x, ScalarType dtype) -> Tensor")
npulib.define("npu_dtype_cast(Tensor x, ScalarType dtype) -> Tensor")
