#ifdef __CCE_KT_TEST__
#include <stdint.h>
#define BEGIN_TILING_DATA_DEF(name) struct name {
#define TILING_DATA_FIELD_DEF(type, name) \
  type name; \
  inline void set_##name(type value) { name = value; } \
  inline type get_##name() { return name; }
#define END_TILING_DATA_DEF };
#define REGISTER_TILING_DATA_CLASS(op_type, tiling_type)
#else
#include "register/tilingdata_base.h"
#endif

namespace optiling {
    BEGIN_TILING_DATA_DEF(TilingData)
        TILING_DATA_FIELD_DEF(uint32_t, block_dim);
        TILING_DATA_FIELD_DEF(uint32_t, s0);
        TILING_DATA_FIELD_DEF(uint32_t, s1);
        TILING_DATA_FIELD_DEF(uint32_t, s2);
        TILING_DATA_FIELD_DEF(uint32_t, z0b_size);
        TILING_DATA_FIELD_DEF(uint32_t, z1t_size);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(LoadAbsStore, TilingData)
}

