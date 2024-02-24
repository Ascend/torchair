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

#ifndef FUSION_ENGINE_INC_COMMON_FE_LOG_H_
#define FUSION_ENGINE_INC_COMMON_FE_LOG_H_

#include <sys/syscall.h>
#include <unistd.h>
#include <securec.h>
#include <cstdint>
#include <string>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "toolchain/slog.h"
#include "common/util/error_manager/error_manager.h"
#include "common/fe_error_code.h"

/** Assigned FE name in log */
const std::string FE_MODULE_NAME = "FE";

inline uint64_t FeGetTid() {
  thread_local static uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
  return tid;
}

#define D_FE_LOGD(MOD_NAME, fmt, ...) dlog_debug(FE, "%lu %s:" #fmt "\n", FeGetTid(), __FUNCTION__, ##__VA_ARGS__)
#define D_FE_LOGI(MOD_NAME, fmt, ...) dlog_info(FE, "%lu %s:" #fmt "\n", FeGetTid(), __FUNCTION__, ##__VA_ARGS__)
#define D_FE_LOGW(MOD_NAME, fmt, ...) dlog_warn(FE, "%lu %s:" #fmt "\n", FeGetTid(), __FUNCTION__, ##__VA_ARGS__)
#define D_FE_LOGE(MOD_NAME, fmt, ...) dlog_error(FE, "%lu %s:" #fmt "\n", FeGetTid(), __FUNCTION__, ##__VA_ARGS__)

#define FE_LOGDEBUG(...) D_FE_LOGD(FE_MODULE_NAME, __VA_ARGS__)
#define FE_LOGI(...) D_FE_LOGI(FE_MODULE_NAME, __VA_ARGS__)
#define FE_LOGW(...) D_FE_LOGW(FE_MODULE_NAME, __VA_ARGS__)
#define FE_LOGE(...) D_FE_LOGE(FE_MODULE_NAME, __VA_ARGS__)

constexpr const int FE_MAX_LOG_SIZE = 8192;
constexpr const int FE_MSG_LEN = 1024;
constexpr const size_t FE_MSG_MAX_LEN = FE_MSG_LEN - 200;

#define FE_LOGD(...)                                                      \
  do {                                                                         \
    if (CheckLogLevel(static_cast<int32_t>(FE), DLOG_DEBUG) != 1) {            \
      break;                                                                   \
    }                                                                          \
    char msgbuf[FE_MAX_LOG_SIZE];                                              \
    if (snprintf_s(msgbuf, sizeof(msgbuf), sizeof(msgbuf) - 1, ##__VA_ARGS__) == -1) {     \
      msgbuf[sizeof(msgbuf) - 1] = '\0';                                      \
    }                                                                         \
    size_t msg_len = std::strlen(msgbuf);                                     \
    if (msg_len < FE_MSG_MAX_LEN) {                                           \
      FE_LOGDEBUG(__VA_ARGS__);                                               \
      break;                                                                  \
    }                                                                         \
    char *msg_chunk_begin = msgbuf;                                           \
    char *msg_chunk_end = nullptr;                                            \
    while (msg_chunk_begin < msgbuf + msg_len) {                              \
      if (msg_chunk_begin[0] == '\n') {                                       \
        FE_LOGDEBUG("");                                                      \
        msg_chunk_begin += 1;                                                 \
        continue;                                                             \
      }                                                                       \
      msg_chunk_end = std::strchr(msg_chunk_begin, '\n');                     \
      if (msg_chunk_end == nullptr) {                                         \
        msg_chunk_end = msg_chunk_begin + std::strlen(msg_chunk_begin);       \
      }                                                                       \
      while (msg_chunk_end > msg_chunk_begin) {                               \
        std::string msg_chunk(msg_chunk_begin,                                \
                              std::min(FE_MSG_MAX_LEN, static_cast<size_t>(msg_chunk_end - msg_chunk_begin))); \
        FE_LOGDEBUG("%s", msg_chunk.c_str());                                 \
        msg_chunk_begin += msg_chunk.size();                                  \
      }                                                                       \
      msg_chunk_begin += 1;                                                   \
    }                                                                         \
  } while (0)

#define IsDebugLogLevel (CheckLogLevel(FE, DLOG_DEBUG) == 1)

#define FE_LOGD_IF(cond, ...) \
  if ((cond)) {               \
    FE_LOGD(__VA_ARGS__);     \
  }

#define FE_LOGI_IF(cond, ...) \
  if ((cond)) {               \
    FE_LOGI(__VA_ARGS__);     \
  }

#define FE_LOGW_IF(cond, ...) \
  if ((cond)) {               \
    FE_LOGW(__VA_ARGS__);     \
  }

#define FE_LOGE_IF(cond, ...) \
  if ((cond)) {               \
    FE_LOGE(__VA_ARGS__);     \
  }

#define FE_CHECK(cond, log_func, return_expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      return_expr;                            \
    }                                         \
  } while (0)

// If failed to make_shared, then print log and execute the statement.
#define FE_MAKE_SHARED(exec_expr0, exec_expr1) \
  do {                                         \
    try {                                      \
      exec_expr0;                              \
    } catch (...) {                            \
      FE_LOGE("Make shared failed");           \
      exec_expr1;                              \
    }                                          \
  } while (0)

#define FE_CHECK_NOTNULL(val)                           \
  do {                                                  \
    if ((val) == nullptr) {                             \
      FE_LOGE("Parameter[%s] must not be null.", #val); \
      return fe::PARAM_INVALID;                         \
    }                                                   \
  } while (0)

#define FE_CHECK_NOTNULL_WARNLOG(val)                   \
  do {                                                  \
    if ((val) == nullptr) {                             \
      FE_LOGW("Parameter[%s] must not be null.", #val); \
      return fe::PARAM_INVALID;                         \
    }                                                   \
  } while (0)

#define REPORT_FE_ERROR(fmt, ...)  \
  do {                                                  \
    REPORT_INNER_ERROR(EM_INNER_ERROR, fmt, ##__VA_ARGS__);     \
    FE_LOGE(fmt, ##__VA_ARGS__);                        \
  } while (0)

#define REPORT_FE_WARN(fmt, ...)  \
  do {                                                  \
    REPORT_INNER_ERROR(EM_INNER_WARN, fmt, ##__VA_ARGS__);     \
    FE_LOGW(fmt, ##__VA_ARGS__);                        \
  } while (0)
#endif  // FUSION_ENGINE_INC_COMMON_FE_LOG_H_
