#include <ATen/EmptyTensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cpu/Loops.h>

#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/csrc/Device.h>
#include <torch/extension.h>

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, c10::GetDefaultCPUAllocator());

at::Tensor privateuse1_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
                                           c10::optional<at::Layout> layout, c10::optional<at::Device> device,
                                           c10::optional<bool> pin_memory,
                                           c10::optional<at::MemoryFormat> memory_format) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, c10::GetDefaultCPUAllocator(), private_use_ks, c10::dtype_or_default(dtype),
                                   memory_format);
}

at::Tensor &privateuse1_fill__scalar(at::Tensor &self, const at::Scalar &value) {
  return self;
}

at::Tensor privateuse1__copy_from(const at::Tensor &self, const at::Tensor &dst, bool non_blocking) {
  return dst;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &privateuse1_empty_memory_format);
  m.impl("fill_.Scalar", &privateuse1_fill__scalar);
  m.impl("_copy_from", &privateuse1__copy_from);
}

c10::Device device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

PYBIND11_MODULE(_privateuse1_backend, m) {
  m.def("device", &device, "get device name");
}