/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "utils_tools.h"
#include <dlfcn.h>
#include <chrono>
#include <future>
#include <utility>

#include "checker.h"
#include "logger.h"

namespace {
inline const char *GetOpApiLibName(void)
{
    return "libopapi.so";
}

inline void *GetOpApiFuncAddrInLib(void *handler, const char *lib_name, const char *api_name)
{
    auto func_addr = dlsym(handler, api_name);
    if (func_addr == nullptr) {
        TNG_LOG(INFO) << "dlsym "<< api_name << "from" << lib_name <<
            " failed, error:" << dlerror() <<".";
    }
    return func_addr;
}

inline void *GetOpApiLibHandler(const char *lib_name)
{
    auto handler = dlopen(lib_name, RTLD_LAZY);
    if (handler == nullptr) {
        TNG_LOG(ERROR) << "dlopen "<< lib_name << " failed, error: " <<
            dlerror() << ".";
    }
    return handler;
}

}
namespace tng {

bool NpuOpUtilsTools::CheckAclnnAvaliable(const std::string &aclnn_name) {
    static auto opapi_handler = GetOpApiLibHandler(GetOpApiLibName());
    if (opapi_handler != nullptr) {
        auto func_addr = GetOpApiFuncAddrInLib(opapi_handler, GetOpApiLibName(), aclnn_name.c_str());
        if (func_addr != nullptr) {
            return true;
        }
    }
    return false;
}

} // namespace tng