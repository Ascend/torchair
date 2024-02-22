#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "load_broadcast_store_tiling.h"

extern "C" __global__ __aicore__ void load_broadcast_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(optiling::TilingData& tiling_data);

class E2E_Load_Broadcast_Store : public testing::Test,
                               public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_Load_Broadcast_Store, BroadCastLast111) {
  uint32_t block_dim = 48;
  uint32_t s0 = GetParam()[0];
  uint32_t s1 = GetParam()[1];

  optiling::TilingData tiling_data;
  float* x = (float*)AscendC::GmAlloc(s0 * sizeof(float));
  float* y = (float*)AscendC::GmAlloc(s0*s1 * sizeof(float));
  float* expect = (float*)AscendC::GmAlloc(s0*s1 * sizeof(float));

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

INSTANTIATE_TEST_SUITE_P(M32_K_BlockAlign, E2E_Load_Broadcast_Store,
   ::testing::Values(
       std::vector<int>{48*4*8, 16},
       std::vector<int>{48*4*16, 16},
       std::vector<int>{48*4*32, 16},
       std::vector<int>{48*4*64, 16},
       std::vector<int>{48*4*128, 16},
       std::vector<int>{48*4*256, 16},
       std::vector<int>{48*4*488, 16},
       std::vector<int>{48*4*8, 976},
       std::vector<int>{48*4*8, 512},
       std::vector<int>{48*4*8, 256},
       std::vector<int>{48*4*8, 128},
       std::vector<int>{48*4*8, 64},
       std::vector<int>{48*4*8, 32}
   ));