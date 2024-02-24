/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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


#ifndef COMMON_GRAPH_UTILS_FILE_UTILS_H_
#define COMMON_GRAPH_UTILS_FILE_UTILS_H_

#include <cstdint>
#include <string>
#include "external/graph/types.h"
#include "graph/ge_error_codes.h"
#include "ge_common/ge_api_types.h"

namespace ge {

/**
 * @ingroup domi_common
 * @brief  Absolute path for obtaining files
 * @param [in] path of input file
 * @return string. Absolute path of a file. If the absolute path cannot be
 * obtained, an empty string is returned
 */
std::string RealPath(const char_t *path);

/**
 * @ingroup domi_common
 * @brief  Recursively Creating a Directory
 * @param [in] directory_path  Path, which can be a multi-level directory.
 * @return 0 success, 1- fail.
 */
int32_t CreateDir(const std::string &directory_path);

/**
 * @ingroup domi_common
 * @brief  Recursively Creating a Directory with mode
 * @param [in] directory_path  Path, which can be a multi-level directory.
 * @param [in] mode  dir mode, E.G., 0700
 * @return 0 success, 1- fail.
 */
int32_t CreateDir(const std::string &directory_path, uint32_t mode);

/**
 * @ingroup domi_common
 * @brief  Recursively Creating a Directory, deprecated, use CreateDir instead
 * @param [in] directory_path  Path, which can be a multi-level directory.
 * @return 0 success, 1- fail.
 */
int32_t CreateDirectory(const std::string &directory_path);

std::unique_ptr<char[]> GetBinFromFile(std::string &path, uint32_t &data_len);

graphStatus WriteBinToFile(std::string &path, char_t *data, uint32_t &data_len);

/**
 * @ingroup domi_common
 * @brief  Get binary file from file
 * @param [in] name origin name.
 * @return string. name which repace special code as _.
 */
std::string GetRegulatedName(const std::string name);

/**
 * @ingroup domi_common
 * @brief  Get binary file from file
 * @param [in] path  file path.
 * @param [out] buffer char[] used to store file data
 * @param [out] data_len store read size
 * @return graphStatus GRAPH_SUCCESS: success, OTHERS: fail.
 */
graphStatus GetBinFromFile(const std::string &path, char_t *buffer, size_t &data_len);

/**
 * @ingroup domi_common
 * @brief  Write binary to file
 * @param [in] fd  file desciption.
 * @param [in] data char[] used to write to file
 * @param [in] data_len store write size
 * @return graphStatus GRAPH_SUCCESS: success, OTHERS: fail.
 */
graphStatus WriteBinToFile(const int32_t fd, const char_t * const data, size_t data_len);

/**
 * @ingroup domi_common
 * @brief  Save data to file
 * @param [in] file_path  file path.
 * @param [in] data char[] used to store file data
 * @param [in] length store read size
 * @return graphStatus GRAPH_SUCCESS: success, OTHERS: fail.
 */
graphStatus SaveBinToFile(const char * const data, size_t length, const std::string &file_path);

/**
 * @ingroup domi_common
 * @brief  split file path to directory path and file name
 * @param [in] file_path  file path.
 * @param [out] dir_path directory path
 * @param [out] file_name file name
 * @return graphStatus GRAPH_SUCCESS: success, OTHERS: fail.
 */
void SplitFilePath(const std::string &file_path, std::string &dir_path, std::string &file_name);

/**
 * @ingroup domi_common
 * @brief  Get ASCEND_WORK_PATH environment variable
 * @param [out] ascend_work_path ASCEND_WORK_PATH's value.
 * @return graphStatus SUCCESS: success, OTHERS: fail.
 */
Status GetAscendWorkPath(std::string &ascend_work_path);
}

#endif // end COMMON_GRAPH_UTILS_FILE_UTILS_H_
