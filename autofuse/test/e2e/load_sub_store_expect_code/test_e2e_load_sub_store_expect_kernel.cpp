#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "load_sub_store_tiling.h"
extern "C" __global__ __aicore__ void load_sub_store(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(optiling::TilingData& tiling_data);


TEST(E2E_LoadSubStore_Code, CalculateCorrect_Sub)
{
    uint32_t block_dim = 48;
    int test_shape[3] = {96, 16, 16};
    int test_size = test_shape[0] * test_shape[1] * test_shape[2];

    optiling::TilingData tiling_data;
    half* x1 = (half*)AscendC::GmAlloc(test_size * sizeof(half));
    half* x2 = (half*)AscendC::GmAlloc(test_size * sizeof(half));
    half* y = (half*)AscendC::GmAlloc(test_size * sizeof(half));
    half* expect = (half*)AscendC::GmAlloc(test_size * sizeof(half));

    // Prepare test and expect data
    for (int i = 0; i < test_size; i++) {
        x1[i] = i*(-2);
        x2[i] = i*(-1);
        expect[i] = x1[i] - x2[i];
    }

    // Launch
    tiling_data.block_dim = block_dim;
    tiling_data.s0 = test_shape[0];
    tiling_data.s1 = test_shape[1];
    tiling_data.s2 = test_shape[2];
    GetTiling(tiling_data);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(load_sub_store, tiling_data.block_dim, (uint8_t*)x1, (uint8_t*)x2, (uint8_t*)y, nullptr, (uint8_t*)&tiling_data);

    // Count difference
    uint32_t diff_count = 0;
    for (int i = 0; i < test_size; i++) {
        half diff = y[i] - expect[i];
        if (diff > (half)0.0001 || diff < (half)-0.0001) {
            diff_count++;
        }
    }

    EXPECT_EQ(diff_count, 0) << " of " << test_size;

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(expect);
}
