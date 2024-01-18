#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "load_rmax_store_tiling.h"
extern "C" __global__ __aicore__ void load_rmax_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(optiling::TilingData& tiling_data);

class E2E_LoadRmaxStore_Code : public testing::Test,
    public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_LoadRmaxStore_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  optiling::TilingData tiling_data;
  half *x = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);
  half *y = (half *)AscendC::GmAlloc(test_shape[1] * sizeof(half) + 32);
  half *expect = (half *)AscendC::GmAlloc(test_shape[1] * sizeof(half) + 32);

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[1]; j++) {
      int offset = i * test_shape[1] + j;
      x[offset] = offset;
    }
    expect[i] = x[(i + 1) * test_shape[1] - 1];
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_rmax_store, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr, (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_shape[0]; i++) {
    half diff = y[i] - expect[i];
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadRmaxStore_Code,
    ::testing::Values(
        std::vector<int>{96*2, 32},
        std::vector<int>{96*2, 128},
        std::vector<int>{96*2, 256}));
