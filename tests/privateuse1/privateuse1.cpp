#include <ATen/EmptyTensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cpu/Loops.h>

#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/csrc/Device.h>
#include <torch/extension.h>

static c10::DeviceIndex npu_device_index = 0;

struct DummyNpuAllocator final : at::Allocator {
  DummyNpuAllocator() = default;

  at::DataPtr allocate(size_t nbytes) const override {
    void *data = c10::alloc_cpu(nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, npu_device_index)};
  }

  static void ReportAndDelete(void *ptr) {
    if (!ptr) {
      return;
    }
    c10::free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }
};

// Register npu dummy allocator with device type npu
static DummyNpuAllocator global_npu_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_npu_alloc);

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

at::Tensor npu_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
                             c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                             c10::optional<bool> pin_memory_opt) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);

  auto dtype = c10::dtype_or_default(dtype_opt);
  auto res_tensor = at::detail::empty_strided_generic(size, stride, &global_npu_alloc, private_use_ks, dtype);
  return res_tensor;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &privateuse1_empty_memory_format);
  m.impl("fill_.Scalar", &privateuse1_fill__scalar);
  m.impl("_copy_from", &privateuse1__copy_from);
  m.impl("empty_strided", &npu_empty_strided);
}

c10::Device get_npu_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

PYBIND11_MODULE(_privateuse1_backend, m) {
  m.def("npu_device", &get_npu_device, "get npu device object");
}
