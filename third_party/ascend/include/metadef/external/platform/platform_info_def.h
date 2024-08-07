/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef PLATFORM_INFO_DEF_H
#define PLATFORM_INFO_DEF_H

#include <map>
#include <string>
#include <vector>

using std::map;
using std::vector;
using std::string;

namespace fe {
enum MemoryType { DDR = 0, HBM };

enum L2Type { Cache = 0, Buff };

enum JitCompileMode { REUSE_BINARY = 0, COMPILE_ONLINE, AUTO };

typedef struct tag_str_info {
  std::string aic_version;
  std::string ccec_aic_version;
  std::string ccec_aiv_version;
  std::string is_support_ai_cpu_compiler;
  std::string short_soc_version;
} StrInfo;

typedef struct tag_so_c_info {
  uint32_t ai_core_cnt;
  uint32_t vector_core_cnt;
  uint32_t ai_cpu_cnt;
  MemoryType memory_type;
  uint64_t memory_size;
  L2Type l2_type;
  uint64_t l2_size;
  uint32_t l2PageNum;
  uint32_t task_num;
  int32_t arch_type;
  int32_t chip_type;
  tag_so_c_info()
      : ai_core_cnt(0), vector_core_cnt(0), memory_size(0), l2_size(0), l2PageNum(0), task_num(0), arch_type(-1),
        chip_type(-1) {}
} SoCInfo;

typedef struct tag_ai_core_spec {
  double cube_freq;
  uint64_t cube_m_size;
  uint64_t cube_n_size;
  uint64_t cube_k_size;
  uint64_t vec_calc_size;
  uint64_t l0_a_size;
  uint64_t l0_b_size;
  uint64_t l0_c_size;
  uint64_t l1_size;
  uint64_t smask_buffer;
  uint64_t ub_size;
  uint64_t ubblock_size;
  uint64_t ubbank_size;
  uint64_t ubbank_num;
  uint64_t ubburst_in_one_block;
  uint64_t ubbank_group_num;
  uint32_t unzip_engines;
  uint32_t unzip_max_ratios;
  uint32_t unzip_channels;
  uint8_t unzip_is_tight;
  uint8_t cube_vector_split;
  uint8_t sparsity;
} AiCoreSpec;

typedef struct tag_ai_core_memory_rates {
  double ddr_rate;
  double ddr_read_rate;
  double ddr_write_rate;
  double l2_rate;
  double l2_read_rate;
  double l2_write_rate;
  double l1_to_l0_a_rate;
  double l1_to_l0_b_rate;
  double l1_to_ub_rate;
  double l0_c_to_ub_rate;
  double ub_to_l2_rate;
  double ub_to_ddr_rate;
  double ub_to_l1_rate;
} AiCoreMemoryRates;

typedef struct tag_vector_core_spec {
  double vec_freq;
  uint64_t vec_calc_size;
  uint64_t smask_buffer;
  uint64_t ub_size;
  uint64_t ubblock_size;
  uint64_t ubbank_size;
  uint64_t ubbank_num;
  uint64_t ubburst_in_one_block;
  uint64_t ubbank_group_num;
  uint64_t vector_reg_size;
  uint64_t predicate_reg_size;
  uint64_t address_reg_size;
  uint64_t alignment_reg_size;
} VectorCoreSpec;

typedef struct tag_vector_core_memory_rates {
  double ddr_rate;
  double ddr_read_rate;
  double ddr_write_rate;
  double l2_rate;
  double l2_read_rate;
  double l2_write_rate;
  double ub_to_l2_rate;
  double ub_to_ddr_rate;
} VectorCoreMemoryRates;

typedef struct tag_cpu_cache {
  uint32_t AICPUSyncBySW;
  uint32_t TSCPUSyncBySW;
} CPUCache;

typedef struct tag_software_spec {
  bool jit_compile_default_value;
  JitCompileMode jit_compile_mode;
  tag_software_spec() {
    jit_compile_default_value = false;
    jit_compile_mode = AUTO;
  }
} SoftwareSpec;

typedef struct tag_platform_info {
  StrInfo str_info;
  SoCInfo soc_info;
  AiCoreSpec ai_core_spec;
  AiCoreMemoryRates ai_core_memory_rates;
  std::map<std::string, std::vector<std::string>> ai_core_intrinsic_dtype_map;
  VectorCoreSpec vector_core_spec;
  VectorCoreMemoryRates vector_core_memory_rates;
  CPUCache cpucache;
  std::map<std::string, std::vector<std::string>> vector_core_intrinsic_dtype_map;
  SoftwareSpec software_spec;
} PlatformInfo;

typedef struct tag_optional_info {
  std::string soc_version;
  std::string core_type;
  uint32_t ai_core_num;
  std::string l1_fusion_flag;
} OptionalInfo;
}  // namespace fe
#endif
