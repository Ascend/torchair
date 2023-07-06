#include <iostream>
#include "core/NPUStream.h"

namespace c10_npu {
namespace {
c10::Device device(c10::kCPU, 0);
c10::Stream default_stream = c10::Stream(c10::Stream::Default::DEFAULT, device);
}  // namespace

NPUStream getCurrentNPUStream(c10::DeviceIndex device_index) { return NPUStream(default_stream); }

std::ostream &operator<<(std::ostream &stream, const NPUStream &s) { return stream << s.unwrap(); }

}  // namespace c10_npu
