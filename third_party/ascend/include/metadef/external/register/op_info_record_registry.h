/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_REGISTER_OP_INFO_RECORD_REGISTRY_H_
#define INC_EXTERNAL_REGISTER_OP_INFO_RECORD_REGISTRY_H_
#include <string>

#include "external/exe_graph/runtime/tiling_context.h"

namespace OpInfoRecord {
struct OpCompilerOption {
    explicit OpCompilerOption(const std::string &impl_mode_v, bool deterministic_v = true) :
        impl_mode(impl_mode_v), deterministic(deterministic_v) {}
    explicit OpCompilerOption(const char *impl_mode_v, bool deterministic_v = true) :
        impl_mode(impl_mode_v), deterministic(deterministic_v) {}
    std::string impl_mode;
    bool deterministic;
};

struct OpKernelInfo {
    explicit OpKernelInfo(const std::string &bin_info_v, int8_t bin_type_v) :
        bin_info(bin_info_v), bin_type(bin_type_v) {}
    explicit OpKernelInfo(const char *bin_info_v, int8_t bin_type_v) :
        bin_info(bin_info_v), bin_type(bin_type_v) {}
    std::string bin_info;
    int8_t bin_type;
};

class __attribute__((visibility("default"))) OpInfoRecordRegister {
public:
    using NotifyFn = void(*)(bool);
    static OpInfoRecordRegister *Instance();
    /*
    * @ingroup OpInfoRecord
    * @brief Register the notification function
    * @param notify_fn [IN] Callback notification function.
    */
    void RegNotify(const NotifyFn notifyFn) const;

    /*
    * @ingroup OpInfoRecord
    * @brief Obtains the current switch status
    * @retval true: The switch is enabled.
    * @retval false: The switch is disablesd.
    */
    bool GetSwitchState() const;

    /*
    * @ingroup OpInfoRecord
    * @brief Output the current operator information
    *
    * @param ctx [IN] Operator context information
    * @param opt [IN] Operator compile option
    */
    void ExeOptInfoStat(
        const gert::TilingContext *ctx,
        const OpCompilerOption &opt,
        const OpKernelInfo *kernelInfo) const;

private:
    OpInfoRecordRegister() = default;
    ~OpInfoRecordRegister() = default;
    OpInfoRecordRegister(const OpInfoRecordRegister &) = delete;
    OpInfoRecordRegister &operator=(const OpInfoRecordRegister &) = delete;
};  // class OpInfoRecordRegister
}  // namespace OpInfoRecord
#endif  // INC_EXTERNAL_REGISTER_OP_INFO_RECORD_REGISTRY_H_
