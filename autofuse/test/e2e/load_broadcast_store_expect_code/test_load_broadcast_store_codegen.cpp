#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "load_broadcast_store_tiling.h"

extern "C" __global__ __aicore__ void load_broadcast_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(optiling::TilingData& tiling_data);

TEST(E2E_Load_Broadcast_Store, BroadCastLast_48block_one_32x32tile) {
  uint32_t block_dim = 48;
  uint32_t s0 = 48*32;
  uint32_t s1 = 64;

  optiling::TilingData tiling_data;
  half* x = (half*)AscendC::GmAlloc(s0 * sizeof(half));
  half* y = (half*)AscendC::GmAlloc(s0*s1 * sizeof(half));
  half* expect = (half*)AscendC::GmAlloc(s0*s1 * sizeof(half));

  // Prepare test and expect data
  for (int i = 0; i < s0; i++) {
    x[i] = i;
    for (int j = 0; j < s1; j++) {
      expect[i * s1 + j] = i;
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_broadcast_store, tiling_data.block_dim, (uint8_t*)x, (uint8_t*)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < s0; i++) {
    for (int j = 0; j < s1; j++) {
      if (y[i * s1 + j] != expect[i * s1 + j]) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << std::to_string(s0 * s1);

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}
