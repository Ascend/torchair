#pragma once

#include <cstdint>
#include <mutex>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/SmallVector.h>
#include <c10/util/Exception.h>
#include "acl/acl_op.h"

namespace c10_npu {

class NPUStream {
public:
  enum Unchecked { UNCHECKED };

  explicit NPUStream(c10::Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == c10::DeviceType::PrivateUse1);
  }

  explicit NPUStream(Unchecked, c10::Stream stream) : stream_(stream) {}

  ~NPUStream(){}

  bool operator==(const NPUStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const NPUStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// Implicit conversion to rtStream_t.
  operator aclrtStream() const {
    return stream();
  }

  /// Implicit conversion to pytorch Stream.
  operator c10::Stream() const {
    return unwrap();
  }

  /// Used to avoid baking in device type explicitly to Python-side API.
  c10::DeviceType device_type() const {
    return c10::DeviceType::PrivateUse1;
  }

  /// Get the NPU device index that this stream is associated with.
  c10::DeviceIndex device_index() const {
    return stream_.device_index();
  }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a NPU device.
  c10::Device device() const {
    return c10::Device(c10::DeviceType::PrivateUse1, device_index());
  }

  c10::StreamId id() const {
    return stream_.id();
  }

  void synchronize() const;

  /// Explicit conversion to rtStream_t.
  aclrtStream stream(const bool need_empty = true) const;

  /// Explicit conversion to Stream.
  c10::Stream unwrap() const {
    return stream_;
  }

private:
  c10::Stream stream_;
};

NPUStream getCurrentNPUStream(c10::DeviceIndex device_index = -1);

std::ostream& operator<<(std::ostream& stream, const NPUStream& s);
} // namespace c10_npu

namespace std {
template <>
struct hash<c10_npu::NPUStream> {
  size_t operator()(c10_npu::NPUStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std
