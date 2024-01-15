#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "load_sub_store_tiling.h"
#define GET_TILING_DATA(tiling_data, tiling) \
    optiling::TilingData& tiling_data = *(optiling::TilingData*)(tiling);
#endif

#include "kernel_operator.h"

using namespace AscendC;

namespace utils {
    template <typename T>
    constexpr inline __aicore__ T Max(const T a) {
        return a;
    }

    template <typename T, typename... Ts>
    constexpr inline __aicore__ T Max(const T a, const Ts... ts) {
        return a > Max(ts...) ? a : Max(ts...);
    }

    template <typename T, typename... Ts>
    constexpr inline __aicore__ T Sum(const T a, const Ts... ts) {
        return (a + ... + ts);
    }
}

extern "C" __global__ __aicore__ void load_sub_store(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    int block_dim = GetBlockIdx();
    const int z0B = block_dim % (t.s0 / t.z0b_size); block_dim /= t.s0 / t.z0b_size;
    int all_cores = t.s0 % t.z0b_size == 0 ? t.s0 / t.z0b_size : (t.s0 / t.z0b_size + 1);
    int main_cores = t.s0 / t.z0b_size;
    int main_tail_block_cnt = t.z0b_size;
    if (block_dim >= main_cores) {
        main_tail_block_cnt = t.s0 - main_cores * t.z0b_size;
    }

    GlobalTensor<half> x1_y;
    x1_y.SetGlobalBuffer((__gm__ half*)x1);
    GlobalTensor<half> x2_y;
    x2_y.SetGlobalBuffer((__gm__ half*)x2);
    GlobalTensor<half> store_y;
    store_y.SetGlobalBuffer((__gm__ half*)y);

    TPipe tpipe;

    const uint32_t load1_y_size = (t.z1t_size - 1) * t.s2 + (t.s2 - 1) + 1;
    const uint32_t load1_y_size_tail = 0 * t.s2 + (t.s2 - 1) + 1;
    const uint32_t load1_y_que_depth = 2;
    const uint32_t load1_y_que_buf_num = 2;
    const uint32_t load2_y_size = (t.z1t_size - 1) * t.s2 + (t.s2 - 1) + 1;
    const uint32_t load2_y_size_tail = 0 * t.s2 + (t.s2 - 1) + 1;
    const uint32_t load2_y_que_depth = 2;
    const uint32_t load2_y_que_buf_num = 2;
    const uint32_t sub_y_size = (t.z1t_size - 1) * t.s2 + (t.s2 - 1) + 1;
    const uint32_t sub_y_size_tail = 0 * t.s2 + (t.s2 - 1) + 1;
    const uint32_t sub_y_que_depth = 2;
    const uint32_t sub_y_que_buf_num = 2;


    const uint32_t q0_size = utils::Max(load1_y_size * sizeof(half));
    const uint32_t q0_depth = utils::Max(load1_y_que_depth);
    const uint32_t q0_buf_num = utils::Max(load1_y_que_buf_num);
    TQue<TPosition::VECIN, q0_depth> q0;
    tpipe.InitBuffer(q0, q0_buf_num, q0_size);

    const uint32_t q1_size = utils::Max(load2_y_size * sizeof(half));
    const uint32_t q1_depth = utils::Max(load2_y_que_depth);
    const uint32_t q1_buf_num = utils::Max(load2_y_que_buf_num);
    TQue<TPosition::VECIN, q1_depth> q1;
    tpipe.InitBuffer(q1, q1_buf_num, q1_size);

    const uint32_t q2_size = utils::Max(sub_y_size * sizeof(half));
    const uint32_t q2_depth = utils::Max(sub_y_que_depth);
    const uint32_t q2_buf_num = utils::Max(sub_y_que_buf_num);
    TQue<TPosition::VECOUT, q2_depth> q2;
    tpipe.InitBuffer(q2, q2_buf_num, q2_size);


    for (int z0b = 0; z0b < main_tail_block_cnt; z0b++) {
        for (int z1T = 0; z1T < t.s1 / t.z1t_size; z1T++) {
            {
                LocalTensor<uint8_t> q0_buf = q0.AllocTensor<uint8_t>();
                LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();
                LocalTensor<half> load1_y;
                load1_y.SetAddrWithOffset(q0_buf, 0);
                DataCopy(load1_y[0], x1_y[z0B * (t.s1 * t.s2 * t.z0b_size) + z0b * (t.s1 * t.s2) + z1T * (t.s2 * t.z1t_size)], load1_y_size);
                LocalTensor<half> load2_y;
                load2_y.SetAddrWithOffset(q1_buf, 0);
                DataCopy(load2_y[0], x2_y[z0B * (t.s1 * t.s2 * t.z0b_size) + z0b * (t.s1 * t.s2) + z1T * (t.s2 * t.z1t_size)], load2_y_size);
                q0.EnQue(q0_buf);
                q1.EnQue(q1_buf);
            }
            {
                LocalTensor<uint8_t> q0_buf = q0.DeQue<uint8_t>();
                LocalTensor<uint8_t> q1_buf = q1.DeQue<uint8_t>();
                LocalTensor<uint8_t> q2_buf = q2.AllocTensor<uint8_t>();
                LocalTensor<half> load1_y;
                load1_y.SetAddrWithOffset(q0_buf, 0);
                LocalTensor<half> load2_y;
                load2_y.SetAddrWithOffset(q1_buf, 0);
                LocalTensor<half> sub_y;
                sub_y.SetAddrWithOffset(q2_buf, 0);
                Sub(sub_y[0], load1_y[0], load2_y[0], load1_y_size);
                q2.EnQue(q2_buf);
                q0.FreeTensor(q0_buf);
                q1.FreeTensor(q1_buf);
            }
            {
                LocalTensor<uint8_t> q2_buf = q2.DeQue<uint8_t>();
                LocalTensor<half> sub_y;
                sub_y.SetAddrWithOffset(q2_buf, 0);
                DataCopy(store_y[z0B * (t.s1 * t.s2 * t.z0b_size) + z0b * (t.s1 * t.s2) + z1T * (t.s2 * t.z1t_size)], sub_y[0], sub_y_size);
                q2.FreeTensor(q2_buf);
            }
        }
    }
}
