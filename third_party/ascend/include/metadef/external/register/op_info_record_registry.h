/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
