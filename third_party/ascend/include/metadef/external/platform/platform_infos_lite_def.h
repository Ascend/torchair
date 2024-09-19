/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_PLATFORM_PLATFORM_INFOS_LITE_DEF_H_
#define INC_EXTERNAL_PLATFORM_PLATFORM_INFOS_LITE_DEF_H_
#include "platform_infos_def.h"
namespace fe {
enum class SocVersion {
  /* claster ASCEND910 */
  ASCEND910A = 0,
  ASCEND910B = 1,
  ASCEND910PREMIUMA = 2,
  ASCEND910PROA = 3,
  ASCEND910PROB = 4,

  /* claster ASCEND910B */
  ASCEND910B1 = 5,
  ASCEND910B2 = 6,
  ASCEND910B3 = 7,
  ASCEND910B4 = 8,

  /* claster ASCEND910_93 */
  ASCEND910_9391 = 9,
  ASCEND910_9381 = 10,
  ASCEND910_9372 = 11,

  /* claster ASCEND310P */
  ASCEND310P1 = 12,
  ASCEND310P3 = 13,

  /* Others */
  ASCEND310 = 14,
  ASCEND310B1 = 15,
  ASCEND310M1 = 16,
  ASCEND610 = 17,
  ASCEND610LITE = 18,
  ASCEND615 = 19,
  ASCEND710VIR01 = 20,
  ASCEND710VIR02 = 21,
  ASCEND710VIR04 = 22,
  ASCEND710VIR08 = 23,
  BS9SX1AA = 24,
  HI3796CV300CS = 25,
  HI3796CV300ES = 26,
  OPTG = 27,
  SD3403 = 28,
  TSNSC = 29,
  TSNSE = 30,
  ASCEND031 = 31,
  ASCEND035 = 32,
  ASCEND310B2 = 33,
  ASCEND310B3 = 34,
  ASCEND310B4 = 35,
  ASCEND320 = 36,
  ASCEND920A = 37,
  ASCEND910B2C = 38,
  SOC_VERSION_BOTTOM
};
const size_t TOTAL_SOC_COUNT = static_cast<size_t>(SocVersion::SOC_VERSION_BOTTOM);

const std::map<std::string, SocVersion> SOC_VERSION_STR {
    {"Ascend910A", SocVersion::ASCEND910A},
    {"Ascend910B", SocVersion::ASCEND910B},
    {"Ascend910PremiumA", SocVersion::ASCEND910PREMIUMA},
    {"Ascend910ProA", SocVersion::ASCEND910PROA},
    {"Ascend910ProB", SocVersion::ASCEND910PROB},
    {"Ascend910B1", SocVersion::ASCEND910B1},
    {"Ascend910B2", SocVersion::ASCEND910B2},
    {"Ascend910B3", SocVersion::ASCEND910B3},
    {"Ascend910B4", SocVersion::ASCEND910B4},
    {"Ascend910B2C", SocVersion::ASCEND910B2C},
    {"Ascend910_9391", SocVersion::ASCEND910_9391},
    {"Ascend910_9381", SocVersion::ASCEND910_9381},
    {"Ascend910_9372", SocVersion::ASCEND910_9372},
    {"Ascend920A", SocVersion::ASCEND920A},
    {"Ascend310P1", SocVersion::ASCEND310P1},
    {"Ascend310P3", SocVersion::ASCEND310P3},
    {"Ascend310", SocVersion::ASCEND310},
    {"Ascend310B1", SocVersion::ASCEND310B1},
    {"Ascend310B2", SocVersion::ASCEND310B2},
    {"Ascend310B3", SocVersion::ASCEND310B3},
    {"Ascend310B4", SocVersion::ASCEND310B4},
    {"Ascend310M1", SocVersion::ASCEND310M1},
    {"Ascend320", SocVersion::ASCEND320},
    {"Ascend610", SocVersion::ASCEND610},
    {"Ascend610Lite", SocVersion::ASCEND610LITE},
    {"Ascend615", SocVersion::ASCEND615},
    {"Ascend710Vir01", SocVersion::ASCEND710VIR01},
    {"Ascend710Vir02", SocVersion::ASCEND710VIR02},
    {"Ascend710Vir04", SocVersion::ASCEND710VIR04},
    {"Ascend710Vir08", SocVersion::ASCEND710VIR08},
    {"BS9SX1AA", SocVersion::BS9SX1AA},
    {"Hi3796CV300CS", SocVersion::HI3796CV300CS},
    {"Hi3796CV300ES", SocVersion::HI3796CV300ES},
    {"OPTG", SocVersion::OPTG},
    {"SD3403", SocVersion::SD3403},
    {"TsnsC", SocVersion::TSNSC},
    {"TsnsE", SocVersion::TSNSE},
    {"Ascend031", SocVersion::ASCEND031},
    {"Ascend035", SocVersion::ASCEND035},
};
/* ==============================AICoreMemoryRates================================ */
enum class AICoreMemoryRatesKey {
  DDR_RATE = 0,
  AI_CORE_MEMORY_RATES_KEY_END
};

const std::vector<std::string> AI_CORE_MEMORY_RATES = {
    "ddr_rate",
};

/* ============================AICoreSpecKey====================================== */
enum class AICoreSpecKey {
  UBBLOCK_SIZE = 0,
  AI_CORE_SPEC_KEY_END
};

const std::vector<std::string> AI_CORE_SPEC = {
    "ubblock_size",
};

/* ============================CPUCacheKey======================================= */
enum class CPUCacheKey {
  AICPU_SYNC_BY_SW = 0,
  TSCPU_SYNC_BY_SW = 0,
  CPU_CACHE_KEY_END
};

const std::vector<std::string> CPU_CACHE = {
    "AICPUSyncBySW",
    "TSCPUSyncBySW"
};

/* =============================DSARandom======================================== */
enum class DSARandomKey {
  COUNTER_WORKSPACE_SIZE = 0,
  DSA_RANDOM_KEY_END
};

const std::vector<std::string> DSA_RANDOM = {
    "CounterWorkspaceSize"
};

/* ============================SoCInfo=========================================== */
enum class SocInfoKey {
  AI_CORE_CNT = 0,
  SOC_INFO_KEY_END
};

const std::vector<std::string> SOC_INFO = {
    "ai_core_cnt"
};

/* ============================SoftwareSpec======================================= */
enum class SoftwareSpecKey {
  JIT_COMPILE_DEFAULT_VALUE = 0,
  JIT_COMPILE_MODE,
  SOFTARE_SPEC_KEY_END
};

const std::vector<std::string> SOFTWARE_SPEC = {
    "jit_compile_default_value",
    "jit_compile_mode"
};

/* ============================VectorCoreMemoryRates==================================== */
enum class VectorCoreMemoryRatesKey {
  DDR_RATE = 0,
  VECTOR_CORE_MEMORY_RATES_KEY_END
};

const std::vector<std::string> VECTOR_CORE_MEMORY_RATES = {
    "ddr_rate"
};

/* ============================VectorCoreSpec====================================== */
enum class VectorCoreSpecKey {
  VEC_FREQ = 0,
  VECTOR_CORE_SPEC_KEY_END
};

const std::vector<std::string> VECTOR_CORE_SPEC = {
    "vec_freq"
};

/* ============================Version============================================== */
enum class VersionKey {
  VERSION_KEY_END
};

const std::vector<std::string> VERSION = {
};


/* ============================IntrinsicTypeKey================================== */
enum class IntrinsicTypeKey {
  MMAD = 0,
  INTRINSIC_TYPE_KEY_END
};

const std::vector<string> INTRINSIC_DTYPE_MAP_KEY_NAME = {
    "Intrinsic_mmad" // MMAD = 0
};

const std::vector<std::string> AI_CORE_INTRINSIC_DTYPE_MAP = INTRINSIC_DTYPE_MAP_KEY_NAME;

const std::vector<std::string> VECTOR_CORE_INTRINSIC_DTYPE_MAP = INTRINSIC_DTYPE_MAP_KEY_NAME;

const std::vector<std::string> ALL_LABEL_END = {};
/* ============================IntrinsicAbility================================== */
enum class IntrinsicAbility {
  add = 0,
  anti_quant_add = 1,
  anti_quant_sub = 2,
  b8u2 = 3,
  bb = 4,
  bf16 = 5,
  bf162f32 = 6,
  bf162s32a = 7,
  bf162s32c = 8,
  bf162s32f = 9,
  bf162s32r = 10,
  bf162s32z = 11,
  bs16 = 12,
  bs32 = 13,
  bs8 = 14,
  bu16 = 15,
  bu32 = 16,
  bu8 = 17,
  cast = 18,
  clip_relu = 19,
  deq = 20,
  deqs322f16 = 21,
  dequant = 22,
  f16 = 23,
  f162f16 = 24,
  f162f32 = 25,
  f162s16 = 26,
  f162s16a = 27,
  f162s16c = 28,
  f162s16f = 29,
  f162s16r = 30,
  f162s16z = 31,
  f162s32a = 32,
  f162s32c = 33,
  f162s32f = 34,
  f162s32r = 35,
  f162s32z = 36,
  f162s4 = 37,
  f162s8 = 38,
  f162s8a = 39,
  f162s8c = 40,
  f162s8f = 41,
  f162s8r = 42,
  f162s8z = 43,
  f162u8 = 44,
  f162u8a = 45,
  f162u8c = 46,
  f162u8f = 47,
  f162u8r = 48,
  f162u8z = 49,
  f16f16 = 50,
  f16f16f16 = 51,
  f16f16s4 = 52,
  f16f16s8 = 53,
  f16f16u16 = 54,
  f16f16u2 = 55,
  f16f32 = 56,
  f16s16 = 57,
  f16s32 = 58,
  f16s32s32 = 59,
  f16s8 = 60,
  f16u16 = 61,
  f16u16f16 = 62,
  f16u2 = 63,
  f16u8 = 64,
  f32 = 65,
  f322bf16 = 66,
  f322bf16a = 67,
  f322bf16c = 68,
  f322bf16f = 69,
  f322bf16r = 70,
  f322bf16z = 71,
  f322f16 = 72,
  f322f16a = 73,
  f322f16c = 74,
  f322f16f = 75,
  f322f16o = 76,
  f322f16r = 77,
  f322f16z = 78,
  f322f32 = 79,
  f322f32a = 80,
  f322f32c = 81,
  f322f32f = 82,
  f322f32r = 83,
  f322f32z = 84,
  f322s16a = 85,
  f322s16c = 86,
  f322s16f = 87,
  f322s16r = 88,
  f322s16z = 89,
  f322s32a = 90,
  f322s32c = 91,
  f322s32f = 92,
  f322s32r = 93,
  f322s32z = 94,
  f322s4 = 95,
  f322s64a = 96,
  f322s64c = 97,
  f322s64f = 98,
  f322s64r = 99,
  f322s64z = 100,
  f322s8 = 101,
  f322u8 = 102,
  f32f16 = 103,
  f32f16f16 = 104,
  f32f32 = 105,
  f32f32f32 = 106,
  f32s16 = 107,
  f32s32 = 108,
  f32u32 = 109,
  f32u32f32 = 110,
  float16 = 111,
  float32 = 112,
  h322f32 = 113,
  int16 = 114,
  int32 = 115,
  int64 = 116,
  int8 = 117,
  normal_relu = 118,
  nz2nd = 119,
  post_act = 120,
  post_eltwise = 121,
  post_quant = 122,
  post_transform = 123,
  pre_act = 124,
  pre_conv = 125,
  quant = 126,
  requant = 127,
  s16 = 128,
  s162f16 = 129,
  s162f16a = 130,
  s162f16c = 131,
  s162f16f = 132,
  s162f16r = 133,
  s162f16z = 134,
  s162f32 = 135,
  s162s16 = 136,
  s162s32 = 137,
  s162s8 = 138,
  s162u32 = 139,
  s162u8 = 140,
  s16f16 = 141,
  s16f32 = 142,
  s16s16 = 143,
  s16s16u16 = 144,
  s16s32 = 145,
  s16s48 = 146,
  s16s64 = 147,
  s16s8 = 148,
  s16u16 = 149,
  s16u16s16 = 150,
  s16u16s8 = 151,
  s16u32 = 152,
  s16u8 = 153,
  s24s16 = 154,
  s24s8 = 155,
  s24u16 = 156,
  s24u8 = 157,
  s32 = 158,
  s322f16 = 159,
  s322f32 = 160,
  s322f32a = 161,
  s322f32c = 162,
  s322f32f = 163,
  s322f32r = 164,
  s322f32z = 165,
  s322s16 = 166,
  s322s4 = 167,
  s322s64 = 168,
  s322s8 = 169,
  s322u16 = 170,
  s322u8 = 171,
  s32f32 = 172,
  s32s16 = 173,
  s32s32 = 174,
  s32s4s4 = 175,
  s32s8s8 = 176,
  s32u16 = 177,
  s32u32 = 178,
  s32u32s32 = 179,
  s32u8 = 180,
  s32u8s8 = 181,
  s32u8u2 = 182,
  s32u8u8 = 183,
  s4 = 184,
  s48s16 = 185,
  s48s32 = 186,
  s48u16 = 187,
  s642f32a = 188,
  s642f32c = 189,
  s642f32f = 190,
  s642f32r = 191,
  s642f32z = 192,
  s642s32 = 193,
  s64s32 = 194,
  s8 = 195,
  s82f16 = 196,
  s82s16 = 197,
  s82s32 = 198,
  s82s8 = 199,
  s8f16 = 200,
  s8f16f16 = 201,
  s8s16 = 202,
  s8s24 = 203,
  s8s32 = 204,
  s8s48 = 205,
  s8s8 = 206,
  s8s8u8 = 207,
  s8u16 = 208,
  scalar_relu = 209,
  sub = 210,
  u16 = 211,
  u162s32 = 212,
  u162u32 = 213,
  u162u8 = 214,
  u16s16 = 215,
  u16s48 = 216,
  u16s64 = 217,
  u16u16 = 218,
  u16u16u16 = 219,
  u16u16u8 = 220,
  u16u32 = 221,
  u16u8 = 222,
  u32 = 223,
  u322s16 = 224,
  u322u16 = 225,
  u322u8 = 226,
  u32s16 = 227,
  u32s32 = 228,
  u32u16 = 229,
  u32u32 = 230,
  u32u32u32 = 231,
  u32u8 = 232,
  u32u8u8 = 233,
  u8 = 234,
  u82f16 = 235,
  u82s16 = 236,
  u82s32 = 237,
  u82u16 = 238,
  u82u32 = 239,
  u8f16 = 240,
  u8f16f16 = 241,
  u8s16 = 242,
  u8s24 = 243,
  u8s48 = 244,
  u8s8 = 245,
  u8u16 = 246,
  u8u32 = 247,
  u8u8 = 248,
  u8u8u8 = 249,
  uint16 = 250,
  uint32 = 251,
  uint64 = 252,
  uint8 = 253,
  vdeqs162b8 = 254,
  vector_relu = 255
};

/* ============================PlatformLabel================================== */
enum class PlatformLabel {
  ENUM_AI_CORE_MEMORY_RATES = 0, /* label string: AICoreMemoryRates */
  ENUM_AI_CORE_SPEC = 1, /* label string: AICoreSpec */
  ENUM_CPU_CACHE = 2, /* label string: CPUCache */
  ENUM_DSA_RANDOM = 3, /* label string: DSARandom */
  ENUM_SOC_INFO = 4, /* label string: SoCInfo */
  ENUM_SOFTWARE_SPEC = 5, /* label string: SoftwareSpec */
  ENUM_VECTOR_CORE_MEMORY_RATES = 6, /* label string: VectorCoreMemoryRates */
  ENUM_VECTOR_CORE_SPEC = 7, /* label string: VectorCoreSpec */
  ENUM_VERSION = 8, /* label string: version */

  /* WARNING: ENUM_AI_CORE_INTRINSIC_DTYPE_MAP must be the first enum among all
   * intrinsic dtype maps. */
  ENUM_AI_CORE_INTRINSIC_DTYPE_MAP = 9, /* label string: AICoreintrinsicDtypeMap */
  ENUM_VECTOR_CORE_INTRINSIC_DTYPE_MAP = 10, /* label string: VectorCoreintrinsicDtypeMap */
  ENUM_ALL_LABEL_END
};
constexpr size_t NORMAL_LABEL_SIZE = static_cast<size_t>(PlatformLabel::ENUM_AI_CORE_INTRINSIC_DTYPE_MAP);
constexpr size_t ALL_LABEL_SIZE = static_cast<size_t>(PlatformLabel::ENUM_ALL_LABEL_END);


struct LabelAndKey {
  LabelAndKey(const string &label_name, const std::vector<std::string> &key_name_vector) {
    pair = std::make_pair(label_name, key_name_vector);
  }

  std::pair<std::string, std::vector<std::string>> pair;
};

enum { COUNTER_BASE = __COUNTER__ };
template<std::size_t COUNTER, std::size_t ENUM>
LabelAndKey MakeQualifiedPair(const string &label_name, const std::vector<std::string> &key_name_vector) {
  static_assert((COUNTER - COUNTER_BASE - 1) == ENUM, "ENUM value is not in the correct sequence");
  return LabelAndKey(label_name, key_name_vector);
}

#define GEN_LABEL(label_name, key_names_vector) \
    GEN_LABEL_UNIQ_HELPER(__COUNTER__, PlatformLabel::ENUM_##key_names_vector, label_name, key_names_vector)

#define GEN_LABEL_UNIQ_HELPER(ctr, enum_label, label_name, key_names_vector) \
    MakeQualifiedPair<ctr, static_cast<size_t>(enum_label)>(label_name, key_names_vector)

const std::vector<LabelAndKey> LABEL_AND_KEYS = {
    GEN_LABEL("AICoreMemoryRates", AI_CORE_MEMORY_RATES),
    GEN_LABEL("AICoreSpec", AI_CORE_SPEC),
    GEN_LABEL("CPUCache", CPU_CACHE),
    GEN_LABEL("DSARandom", DSA_RANDOM),
    GEN_LABEL("SoCInfo", SOC_INFO),
    GEN_LABEL("SoftwareSpec", SOFTWARE_SPEC),
    GEN_LABEL("VectorCoreMemoryRates", VECTOR_CORE_MEMORY_RATES),
    GEN_LABEL("VectorCoreSpec", VECTOR_CORE_SPEC),
    GEN_LABEL("version", VERSION),
    GEN_LABEL("AICoreintrinsicDtypeMap", AI_CORE_INTRINSIC_DTYPE_MAP),
    GEN_LABEL("VectorCoreintrinsicDtypeMap", VECTOR_CORE_INTRINSIC_DTYPE_MAP),
    GEN_LABEL("ALL_LABEL_END", ALL_LABEL_END)
};

/* ============================key name string definition end================================== */
class PlatFormInfosLiteImpl;
using PlatFormInfosLiteImplPtr = std::shared_ptr<PlatFormInfosLiteImpl>;
class PlatFormInfosLite {
 public:
  bool InitPlatFormInfosLite(SocVersion soc_version, PlatFormInfos& old_platform_infos);
  bool GetPlatformRes(PlatformLabel label, uint64_t key, uint64_t &val) const;
  const std::vector<uint64_t> &GetPlatformRes(PlatformLabel label) const;
  bool CheckIntrinsicSupport(IntrinsicTypeKey intrinsic_type, IntrinsicAbility intrinsic_ability) const;
 private:
  PlatFormInfosLiteImplPtr platform_infos_lite_impl_ {nullptr};
};
}
#endif // INC_EXTERNAL_PLATFORM_PLATFORM_INFOS_LITE_DEF_H_
