#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_HDC_CHANNEL_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_HDC_CHANNEL_H_

#include <cstdint>
#include "tng_status.h"

namespace tng {
Status StartStdoutChannel(int32_t device);

void StopStdoutChannel(int32_t device);
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_HDC_CHANNEL_H_
