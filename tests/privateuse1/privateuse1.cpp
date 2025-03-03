#include <ATen/EmptyTensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/StorageImpl.h>
#include <torch/csrc/Device.h>
#include <torch/extension.h>

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(
    PrivateUse1,
    c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);
}
}

static c10::DeviceIndex npu_device_index = 0;

struct DummyNpuAllocator final : at::Allocator {
  DummyNpuAllocator() = default;

#if defined(TNG_TORCH_VERSION) && (TNG_TORCH_VERSION < 20300)  // v2.3.0
  at::DataPtr allocate(size_t nbytes) const override {
#else
  at::DataPtr allocate(size_t nbytes) override {
#endif
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

#if defined(TNG_TORCH_VERSION) && (TNG_TORCH_VERSION >= 20300)  // v2.3.0
  void copy_data(void *dest, const void *src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }
#endif
};

// Register npu dummy allocator with device type npu
static DummyNpuAllocator global_npu_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_npu_alloc);

// Register npu dummy Storage constructor
#if defined(TNG_TORCH_VERSION) && (TNG_TORCH_VERSION < 20300)  // v2.3.0
c10::intrusive_ptr<c10::StorageImpl> make_npu_storage_impl(c10::StorageImpl::use_byte_size_t, c10::SymInt size_bytes,
                                                           c10::Allocator *allocator, bool resizable) {
  c10::intrusive_ptr<c10::StorageImpl> npu_storage_impl =
      c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), size_bytes.as_int_unchecked(),
                                          allocator->allocate(size_bytes.as_int_unchecked()), allocator, resizable);
  return npu_storage_impl;
}
#else
c10::intrusive_ptr<c10::StorageImpl> make_npu_storage_impl(c10::StorageImpl::use_byte_size_t, c10::SymInt size_bytes,
                                                           c10::DataPtr data_ptr, c10::Allocator *allocator,
                                                           bool resizable) {
  c10::intrusive_ptr<c10::StorageImpl> npu_storage_impl;
  if (data_ptr == nullptr) {
    npu_storage_impl =
        c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), size_bytes, allocator, resizable);
  } else {
    npu_storage_impl = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), size_bytes,
                                                             std::move(data_ptr), allocator, resizable);
  }
  return npu_storage_impl;
}
#endif

int npu_storage_register() {
  c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &make_npu_storage_impl);
  return 0;
}

static const int ret = npu_storage_register();

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

at::Tensor npu_scatter_update(const at::Tensor &self, const at::Tensor &indices, const at::Tensor &updates, int64_t axis) {
  return self.clone();
}

at::Tensor &npu_scatter_update_(at::Tensor &self, const at::Tensor &indices, const at::Tensor &updates, int64_t axis) {
  return self;
}

at::Tensor &npu_set_(at::Tensor &self, c10::Storage src, long storage_offset, c10::IntArrayRef size,
                     c10::IntArrayRef stride = {}) {
  void* data_ptr = const_cast<void*>(src.data());
  auto tensor = at::from_blob(data_ptr, size, src.device());
  self = tensor;
  return self;
}

at::Tensor &npu_normal_(at::Tensor &self, const double mean, const double std, const c10::optional<at::Generator>) {
  auto res = self.to(c10::Device(c10::DeviceType::PrivateUse1, 0));
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &privateuse1_empty_memory_format);
  m.impl("fill_.Scalar", &privateuse1_fill__scalar);
  m.impl("_copy_from", &privateuse1__copy_from);
  m.impl("empty_strided", &npu_empty_strided);
  m.impl("scatter_update", &npu_scatter_update);
  m.impl("scatter_update_", &npu_scatter_update_);
  m.impl("set_.source_Storage_storage_offset", &npu_set_);
  m.impl("normal_", &npu_normal_);
}

// Register fallthrough for Autograd backends dispatch keys
// NB: But not the private use ones; maybe the extension wants
// to override it themselves!
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_FRAGMENT(aten, m) {
  m.def("scatter_update(Tensor self, Tensor indices, Tensor updates, int axis) -> Tensor",
        TORCH_FN(npu_scatter_update));
  m.def("scatter_update_(Tensor(a!) self, Tensor indices, Tensor updates, int axis) -> Tensor(a!)",
        TORCH_FN(npu_scatter_update_));
}

c10::Device get_npu_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

class PrivateGeneratorImpl : public at::CPUGeneratorImpl {
public:
    PrivateGeneratorImpl(c10::DeviceIndex device_index) {
      device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
      key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
    }

    ~PrivateGeneratorImpl() override = default;

    void set_offset(uint64_t offset) override {
      offset_ = offset;
    };

    uint64_t get_offset() const override {
      return offset_;
    };

    uint64_t offset_ = 0;
};

// this is used to register generator
at::Generator make_generator_privateuse1(c10::DeviceIndex device_index) {
  return at::make_generator<PrivateGeneratorImpl>(device_index);
}

void register_generator() {
  REGISTER_GENERATOR_PRIVATEUSE1(make_generator_privateuse1)
}

struct FooHooksInterface : public at::PrivateUse1HooksInterface {
    ~FooHooksInterface() override = default;

    const at::Generator &getDefaultGenerator(c10::DeviceIndex device_index) override {
      static auto device_gen = make_generator_privateuse1(device_index);
      return device_gen;
    }
    at::Device getDeviceFromPtr(void* data) const override {
      return at::Device(c10::DeviceType::PrivateUse1, 0);
   }
};

struct FooHooksArgs : public at::PrivateUse1HooksArgs {
};

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, FooHooksInterface, FooHooksArgs
);
#define REGISTER_PRIVATEUSE1_HOOKS(clsname) \
  C10_REGISTER_CLASS(PrivateUse1HooksRegistry, clsname, clsname)

C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, FooHooksInterface, FooHooksArgs
)

static at::PrivateUse1HooksInterface *get_private_hooks() {
  static at::PrivateUse1HooksInterface *privateuse1_hooks;
  static c10::once_flag once;
  c10::call_once(once, [] {
      privateuse1_hooks = PrivateUse1HooksRegistry()->Create("PrivateUse1Hooks", {}).release();
      if (!privateuse1_hooks) {
        privateuse1_hooks = new FooHooksInterface();
      }
  });
  return privateuse1_hooks;
}

void register_hook() {
  at::RegisterPrivateUse1HooksInterface(get_private_hooks());
}

const at::Generator &default_generator(c10::DeviceIndex device_index) {
  return at::globalContext().defaultGenerator(at::Device(c10::DeviceType::PrivateUse1, device_index));
}

PYBIND11_MODULE(_privateuse1_backend, m) {
  m.def("npu_device", &get_npu_device, "get npu device object");
  m.def("register_generator", &register_generator, "register generator for custom device");
  m.def("register_hook", &register_hook, "register_hook for privateuse1");
  m.def("default_generator", &default_generator, "default_generator for privateuse1");
}
