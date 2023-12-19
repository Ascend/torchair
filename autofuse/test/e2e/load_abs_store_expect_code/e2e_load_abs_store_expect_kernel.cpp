#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "load_abs_store_tiling.h"
#define GET_TILING_DATA(tiling_data, tiling) \
    optiling::TilingData& tiling_data = *(optiling::TilingData*)(tiling);
#endif

#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void load_abs_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);

    int z0B = GetBlockIdx() % (t.s0/t.z0b_size);

    GlobalTensor<half> xGm;
    xGm.SetGlobalBuffer((__gm__ half*)x, t.s0*t.s1*t.s2);
    GlobalTensor<half> yGm;
    yGm.SetGlobalBuffer((__gm__ half*)y, t.s0*t.s1*t.s2);

    size_t x_y_size = (t.z1t_size-1)*t.s2 + (t.s2-1)*1 + 1;
    size_t load_y_size = (t.z1t_size-1)*t.s2 + (t.s2-1)*1 + 1;
    size_t abs_y_size = (t.z1t_size-1)*t.s2 + (t.s2-1)*1 + 1;
    size_t store_y_size = (t.z1t_size-1)*t.s2 + (t.s2-1)*1 + 1;

    size_t que0_size = load_y_size;
    size_t que1_size = abs_y_size;

    TPipe tpipe;
    TQue<TPosition::VECIN, 2> que0;
    tpipe.InitBuffer(que0, 2, que0_size*sizeof(half));
    TQue<TPosition::VECOUT, 2> que1;
    tpipe.InitBuffer(que1, 2, que1_size*sizeof(half));

    for (int z0b = 0; z0b < t.z0b_size; z0b++) {
        for (int z1T = 0; z1T < t.s1/t.z1t_size; z1T++) {
            {
                auto x_y = xGm[z0B*t.s1*t.s2*t.z0b_size + z0b*t.s1*t.s2 + z1T*t.s2*t.z1t_size];
                auto load_y = que0.AllocTensor<half>();
                DataCopy(load_y, x_y, x_y_size);
                que0.EnQue(load_y);
            }
            {
                auto load_y = que0.DeQue<half>();
                auto abs_y = que1.AllocTensor<half>();
                Abs(abs_y, load_y, load_y_size);
                que1.EnQue(abs_y);
                que0.FreeTensor(load_y);
            }
            {
                auto abs_y = que1.DeQue<half>();
                auto store_y = yGm[z0B*t.s1*t.s2*t.z0b_size + z0b*t.s1*t.s2 + z1T*t.s2*t.z1t_size];
                DataCopy(store_y, abs_y, abs_y_size);
                que1.FreeTensor(abs_y);
            }
        }
    }
}
