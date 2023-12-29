#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "load_abs_store_tiling.h"
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

extern "C" __global__ __aicore__ void load_abs_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
GET_TILING_DATA(t, tiling);

int block_dim = GetBlockIdx();
const int z0B = block_dim % (t.s0 / t.z0b_size); block_dim /= t.s0 / t.z0b_size;

GlobalTensor<half> x_y;
x_y.SetGlobalBuffer((__gm__ half*)x);
GlobalTensor<half> store_y;
store_y.SetGlobalBuffer((__gm__ half*)y);

TPipe tpipe;

const uint32_t load_y_size = (t.z1t_size - 1) * t.s2 + (t.s2 - 1) + 1;
const uint32_t load_y_que_depth = 2;
const uint32_t load_y_que_buf_num = 2;
const uint32_t abs_y_size = (t.z1t_size - 1) * t.s2 + (t.s2 - 1) + 1;
const uint32_t abs_y_que_depth = 2;
const uint32_t abs_y_que_buf_num = 2;


const uint32_t q0_size = utils::Max(load_y_size * sizeof(half));
const uint32_t q0_depth = utils::Max(load_y_que_depth);
const uint32_t q0_buf_num = utils::Max(load_y_que_buf_num);
TQue<TPosition::VECIN, q0_depth> q0;
tpipe.InitBuffer(q0, q0_buf_num, q0_size);

const uint32_t q1_size = utils::Max(abs_y_size * sizeof(half));
const uint32_t q1_depth = utils::Max(abs_y_que_depth);
const uint32_t q1_buf_num = utils::Max(abs_y_que_buf_num);
TQue<TPosition::VECOUT, q1_depth> q1;
tpipe.InitBuffer(q1, q1_buf_num, q1_size);

    for (int z0b = 0; z0b < t.z0b_size; z0b++) {
        for (int z1T = 0; z1T < t.s1/t.z1t_size; z1T++) {
            {
                auto load_y = q0.AllocTensor<half>();
                DataCopy(load_y, x_y[z0B*t.s1*t.s2*t.z0b_size + z0b*t.s1*t.s2 + z1T*t.s2*t.z1t_size], load_y_size);
                q0.EnQue(load_y);
            }
            {
                auto load_y = q0.DeQue<half>();
                auto abs_y = q1.AllocTensor<half>();
                Abs(abs_y, load_y, load_y_size);
                q1.EnQue(abs_y);
                q0.FreeTensor(load_y);
            }
            {
                auto abs_y = q1.DeQue<half>();
                DataCopy(store_y[z0B*t.s1*t.s2*t.z0b_size + z0b*t.s1*t.s2 + z1T*t.s2*t.z1t_size], abs_y, abs_y_size);
                q1.FreeTensor(abs_y);
            }
        }
    }
}
