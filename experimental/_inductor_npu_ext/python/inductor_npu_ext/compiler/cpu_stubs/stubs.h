#ifndef TORCH_NPU_COMPILER_CPU_STUBS_STUBS_H_
#define TORCH_NPU_COMPILER_CPU_STUBS_STUBS_H_

#include <cstdint>
#include <cstddef>
typedef void *aclrtStream;
namespace at {
    struct Storage { void *data() const { return (void*)0x789; } };
    struct Tensor { Storage storage() const { return {}; } };
}
namespace at_npu { namespace native {
    inline at::Tensor allocate_workspace(uint64_t size, aclrtStream stream) {
        (void)size; (void)stream; return {};
    }
}}

namespace c10_npu {
    struct getCurrentNPUStream{
        void *stream(bool = false){ return (void*)0x123; }
    };
    namespace NPUCachingAllocator {
        void *raw_alloc_with_stream(size_t size, void *stream) { return (void*)0x456; }
        void raw_delete(void *ptr) { return; }
    }
}

namespace at_npu { namespace native {
    struct OpCommand {
        template <typename F>
        static void RunOpApiV2(const char *name, F &&fn) { (void)name; fn(); }
    };
}}

#endif // TORCH_NPU_COMPILER_CPU_STUBS_STUBS_H_
