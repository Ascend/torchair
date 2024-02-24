/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_AICORE_UTIL_TYPES_H_
#define FUSION_ENGINE_INC_COMMON_AICORE_UTIL_TYPES_H_

#include <map>
#include <string>
#include <vector>
#include "graph/anchor.h"
#include "graph/op_desc.h"
#include "runtime/kernel.h"

namespace fe {
struct FusionOpSrc {
  uint32_t src_op_id;
  ge::AnchorPtr src_anchor;
  int32_t fusion_src_index;
  int32_t fusion_dst_index;
};

struct FusionOpDst {
  uint32_t dst_op_id;
  ge::AnchorPtr dst_anchor;
};

struct FusionDataFlow {
  std::pair<ge::AnchorPtr, ge::AnchorPtr> edge;
  std::pair<std::string, ge::AnchorPtr> node_dataindex_pair;
};

using L2FusionData_t = struct tag_l2_fusion_data {
  uint32_t l2Index;
  uint64_t l2Addr;
  uint64_t l2PageNum;
};
using L2FusionDataMap_t = std::map<uint64_t, L2FusionData_t>;

const uint32_t L2_MAXDATANUM = 8;
using fe_sm_desc_t = struct tag_fe_sm_desc {
  rtL2Ctrl_t l2ctrl;
  std::string node_name[L2_MAXDATANUM];
  uint8_t output_index[L2_MAXDATANUM];
};

using TaskL2FusionInfo_t = struct TagTaskL2FusionInfo {
  std::string node_name;
  fe_sm_desc_t l2_info;
  L2FusionDataMap_t input;
  L2FusionDataMap_t output;
  uint32_t is_used;
};

using L2FusionInfoPtr = std::shared_ptr<TaskL2FusionInfo_t>;

using ToOpStruct_t = struct ToOpStruct {
  int64_t op_l1_space = 0;
  std::vector<int64_t> op_l1_fusion_type;
  int64_t op_l1_workspace_flag = 0; // for workspace flag
  int64_t op_l1_workspace_size = 0;
  std::vector<std::vector<int64_t>> slice_input_shape;
  std::vector<std::vector<int64_t>> slice_output_shape;
  std::vector<std::vector<int64_t>>
      slice_input_offset; // conv & pooling & ReadSelect
  std::vector<std::vector<int64_t>> slice_output_offset; // WriteSelect
  std::vector<uint32_t> total_shape;
  uint32_t split_index = 0;
  ToOpStruct() {
    // set invalid value for essential variable
    op_l1_space = -1;
    op_l1_workspace_size = -1;
  }
};

enum class BitShift {
  BIT_SHIFT_8 = 8,
  BIT_SHIFT_16 = 16,
  BIT_SHIFT_24 = 24,
  BIT_SHIFT_32 = 32,
  BIT_SHIFT_40 = 40,
  BIT_SHIFT_48 = 48,
};

struct OpCustomizeDtype {
  std::vector<ge::DataType> input_dtypes;
  std::vector<ge::DataType> output_dtypes;
};

enum SlicePattern {
  ELEMENT_WISE = 0,
  ELEMENT_WISE_BROADCAST,
  BROADCAST,
  SLIDING_WINDOW,
  SLIDING_WINDOW_DECONV,
  CUBE_MATMUL,
  SLICE_PATTERN_REDUCE,
  SLICE_PATTERN_RESIZE,
  SLICE_PATTERN_SCATTER,
  SLICE_PATTERN_SEGMENT,
  PATTERN_RESERVED
};

enum AICoreMode {
  FFTS_MODE_NO_FFTS = 0,
  FFTS_MODE_FFTS,
  FFTS_MODE_FFTS_PLUS,
  FFTS_MODE_RESERVED
};

enum class FFTS_SUPPORT_TYPE {
  FFTS_NO_FORCE = 0,
  FFTS_FORCE_MANUAL,
  FFTS_FORCE_AUTO
};

enum OpImplType : int64_t {
  EN_IMPL_CUSTOM_CONSTANT_CCE = 0,   // custom constant op
  EN_IMPL_CUSTOM_TIK,                // custom tik op
  EN_IMPL_CUSTOM_TBE,                // custom tbe op
  EN_IMPL_HW_CONSTANT_CCE,           // Huawei built-in constant op
  EN_IMPL_HW_GENERAL_CCE,            // Huawei built-in cce op
  EN_IMPL_HW_TIK,                    // Huawei built-in tik op
  EN_IMPL_HW_TBE,                    // Huawei built-in tbe op
  EN_IMPL_RL,                        // RL op
  EN_IMPL_PLUGIN_TBE,                // Huawei built-in tbe plugin op
  EN_IMPL_VECTOR_CORE_HW_TBE,        // Huawei built-in tbe op
  EN_IMPL_VECTOR_CORE_CUSTOM_TBE,    // custom tbe op
  EN_IMPL_NON_PERSISTENT_CUSTOM_TBE, // custom tbe op
  EN_IMPL_HW_DSA,                    // Huawei built-in DSA op
  EN_RESERVED,                       // reserved value
  // sub type used for CUSTOM: sub_type = main_type | (priority << 32)
  EN_SUBTYPE_RESERVED = 0xFFFFFFFFFF // reserved value
};

enum AOEOption {
  AOE_OPT_USE_KB = 0,
  AOE_OPT_NOT_USE_KB,
  AOE_OPT_ONLY_USE_KB,
  AOE_OPT_RESERVED
};

struct OptimizeConfig {
  bool enable_superkernel_plus;
  AOEOption aoe_option;
};

struct PassChangeInfo {
  std::string pass_name;
  int32_t recovery_times;
  int32_t rollback_times;
  vector<int64_t> recovery_scope_ids;
};

struct FEOpsStoreInfo {
  int32_t priority;
  std::string fe_ops_store_name;
  OpImplType op_impl_type;
  std::string cfg_file_path;
  std::string op_impl_file_path;
  bool need_pre_compile;
  bool need_compile;
  FEOpsStoreInfo() : priority(0), fe_ops_store_name(), op_impl_type(EN_RESERVED), cfg_file_path(), op_impl_file_path(),
                     need_pre_compile(false), need_compile(false) {}
  FEOpsStoreInfo(int32_t priority_value, const std::string &ops_store_name_value, OpImplType op_impl_type_value,
                 const std::string &cfg_file_path_value, const std::string &op_impl_file_path_value,
                 bool need_pre_compile_value, bool need_compile_value)
                 : priority(priority_value), fe_ops_store_name(ops_store_name_value), op_impl_type(op_impl_type_value),
                   cfg_file_path(cfg_file_path_value), op_impl_file_path(op_impl_file_path_value),
                   need_pre_compile(need_pre_compile_value), need_compile(need_compile_value) {}
  FEOpsStoreInfo(int32_t priority_value, const std::string &ops_store_name_value, OpImplType op_impl_type_value,
                 const std::string &cfg_file_path_value, const std::string &op_impl_file_path_value)
                 : priority(priority_value), fe_ops_store_name(ops_store_name_value), op_impl_type(op_impl_type_value),
                   cfg_file_path(cfg_file_path_value), op_impl_file_path(op_impl_file_path_value),
                   need_pre_compile(false), need_compile(false) {}
};

enum class ISAArchVersion { EN_ISA_ARCH_V100 = 0, EN_ISA_ARCH_V200, EN_ISA_ARCH_V220, EN_ISA_ARCH_V300,
                            EN_ISA_ARCH_V350 };

enum class CubeVecState { CUBE_VEC_UNKNOWN = 0, CUBE_VEC_DECOUPLE, CUBE_VEC_MIX }; // keep for canndev

enum class CubeVecStateNew { CUBE_VEC_UNKNOWN = 0, CUBE_VEC_SPLIT, CUBE_VEC_FUSE };

enum class UBFusionType { FUSION_TYPE_UB = 0, FUSION_TYPE_MIXL2, FUSION_TYPE_NONE, FUSION_TYPE_RESERVED };

enum class AppendArgsMode { NO_ARGS = 0, L2_BUFFER_ARGS = 1, L2_CACHE_ARGS = 999};

enum BufferOptimize { EN_UNKNOWN_OPTIMIZE = 0, EN_OFF_OPTIMIZE, EN_L1_OPTIMIZE, EN_L2_OPTIMIZE };

enum AutoTuneMode { TUNE_MODE_NO_TUNE = 0, TUNE_MODE_AUTO_TUNE, TUNE_MODE_RL_TUNE, TUNE_MODE_AUTO_AND_RL_TUNE };

enum PrecisionPolicy { WHITE = 0, BLACK = 1, GRAY = 2 };

enum class JitCompileCfg { CFG_FALSE = 0, CFG_TRUE = 1, CFG_AUTO = 2};

enum OpPattern {
  OP_PATTERN_OP_KERNEL = 0,
  OP_PATTERN_OP_CUSTOMIZE,
  OP_PATTERN_FORMAT_AGNOSTIC,
  OP_PATTERN_BROADCAST,
  OP_PATTERN_REDUCE,
  OP_PATTERN_RANGE_AGNOSTIC,
  OP_PATTERN_BROADCAST_ENHANCED
};

enum OpParamType { REQUIRED = 0, OPTIONAL, DYNAMIC, RESERVED };

enum OpConstValueDepend { CONST_IGNORE = 0, CONST_REQUIRED, CONST_OPTIONAL };

enum class DynamicCompileStatic { TUNE = 0, COMPILE, NOT_SUPPORT };

enum class DynamicRankType { NOT_SUPPORT = 0, SUPPORT, UPGRADE_TO_SUPPORT };

enum class JitCompile {
  DEFAULT = 0,
  ONLINE,                         // static_true, dynamic_true || true
  REUSE_BINARY,                   // static_true, dynamic_false || false
  STATIC_BINARY_DYNAMIC_ONLINE,   // static_false, dynamic_true
  STATIC_BINARY_DYNAMIC_BINARY    // static_false, dynamic_false
};

enum class RangeLimitType { DEFAULT = 0, LIMITED, UNLIMITED, DYNAMIC };

const std::unordered_set<std::string> kWeightTypes = {"Const", "Constant", "Variable"};

const std::unordered_set<std::string> kConstTypes = {"Const", "Constant"};

const std::unordered_set<std::string> kMixL2PassName = {"TbeConvBnreduceFusionPass",
                                                        "TbeConv2DBackpropElemwiseFusionPass"};

enum class CmoType {
  CMO_TYPE_PREFETCH = 0,
  CMO_TYPE_INVALID,
  CMO_TYPE_BARRIER,
  CMO_TYPE_WRITEBACK,
  CMO_TYPE_BUTT
};

enum class CmoTypeObject {
  INPUT = 0,
  WEIGHT,
  OUTPUT,
  WORKSPACE
};

enum class L2CacheMode {
  DEFAULT = 0,
  RC,
  CMO
};

enum class L2CacheReadMode {
  RM_NONE = -1,
  READ_LAST = 1,
  READ_INVALID = 2,
  NOT_NEED_WRITEBACK = 3
};

enum class FormatModeType {
  FORMAT_MODE_NZNZ = 0,
  FORMAT_MODE_NDND = 1,
  FORMAT_MODE_NDNZ = 2,
  FORMAT_MODE_INVALID,
};

enum class WeightCompressType {
  LOW_SPARSE_COMPRESS = 0,
  HIGH_SPARSE_COMPRESS,
  DISABLE_COMPRESS
};

enum class AclnnSupportType {
  DEFAULT = 0,
  SUPPORT_ACLNN,
  ACLNN_ONLY
};

enum class MultiKernelSupportType {
  DEFAULT = 0,
  MULTI_SUPPORT,
  SINGLE_SUPPORT
};

using CmoAttr = struct CMO_ATTR {
  ge::NodePtr   node;
  CmoTypeObject object;
  int32_t       object_index;
};

using CmoExtraAttr = std::map<std::string, std::vector<CmoAttr>>;

inline bool IsTbe(const OpImplType& impl_type)
{
  OpImplType main_type = static_cast<OpImplType>(impl_type & 0xFFFFFFFF);
  return main_type == EN_IMPL_HW_TBE || main_type == EN_IMPL_VECTOR_CORE_HW_TBE|| main_type == EN_IMPL_CUSTOM_TBE ||
         main_type == EN_IMPL_VECTOR_CORE_CUSTOM_TBE  || main_type == EN_IMPL_NON_PERSISTENT_CUSTOM_TBE ;
}

template <typename Ret, typename T>
Ret GetMainImplType(T a)
{
  return static_cast<Ret>(a & 0xFFFFFFFF);
}
}
#endif  // FUSION_ENGINE_INC_COMMON_AICORE_UTIL_TYPES_H_
