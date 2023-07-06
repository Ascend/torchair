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

#ifndef FUSION_ENGINE_INC_COMMON_FE_ERROR_CODE_H_
#define FUSION_ENGINE_INC_COMMON_FE_ERROR_CODE_H_

#include <string>

namespace fe {

/* args list for error message */
const std::string PARAMETER0          = "parameter0";
const std::string PARAMETER1          = "parameter1";
const std::string PARAMETER2          = "parameter2";
const std::string EM_VALUE          = "value";
const std::string EM_OPTION         = "option";
const std::string EM_AICORE_NUM     = "ai_core_num";
const std::string EM_NEW_OP         = "new_op";
const std::string EM_SRC_OP         = "src_op";
const std::string EM_SRC_FORMAT     = "src_format";
const std::string EM_DEST_OP        = "dest_op";
const std::string EM_DEST_FORMAT    = "dest_format";
const std::string EM_GRAPH_NAME     = "graph_name";
const std::string EM_FILE           = "file";
const std::string EM_ERROR_MSG      = "errmsg";
const std::string EM_PARAM          = "parameter";
const std::string EM_OP_NAME        = "op_name";
const std::string EM_OP_TYPE        = "op_type";
const std::string EM_GRAPH_ID       = "graph_id";
const std::string EM_THREAD_ID      = "thread_id";
const std::string EM_TASK_ID        = "task_id";
const std::string EM_PASS_NAME      = "pass_name";
const std::string EM_PASS_TYPE      = "pass_type";
const std::string EM_ATTR_OR_PATTERN_NAME = "attr_or_pattern_name";
const std::string EM_OPS_STORE_NAME = "ops_store_name";
const std::string EM_CORRECT_VALUE = "correct_value";
const std::string EM_INDEX = "correct_value";
const std::string EM_COMMON_ERROR = "common_error";
const std::string EM_ORIGIN_DTYPE     = "origin_dtype";
const std::string EM_ENV              = "env";
const std::string EM_SITUATION        = "situation";

/* Input parameter[--%s]'s value is empty!
 * parameter
 * */
const std::string EM_INPUT_PARAM_EMPTY = "E10004";

const std::string EM_INNER_ERROR = "E29999";
const std::string EM_INNER_WARN = "W29999";

/* Failed to compile Op [%s]. (optype: [%s])
 * parameter0,parameter1
 * */
const std::string EM_COMPLIE_FAILED = "E20001";

/* Value [%s] for environment variable [%s] is invalid when %s.
 * parameter0,parameter1,parameter2
 * */
const std::string EM_ENVIRONMENT_VARIABLE_FAILED = "E20002";

/* Configuration file [%s] for parameter [%s] has invalid content. Reason: [%s]
 * parameter0,parameter1,parameter2
 * */
const std::string EM_INVALID_CONTENT = "E20003";

/*
 * Failed to run graph fusion pass [%s]. The pass type is [%s]
 * parameter0,parameter1
 * */
const std::string EM_RUN_PASS_FAILED = "E20007";

/* Value [%s] for parameter [%s] is invalid. Reason: %s.
 * parameter0,parameter1,parameter2
 * */
const std::string EM_INPUT_OPTION_INVALID = "E20101";

/* Value [%s] for parameter [aicore_num] is invalid. The value must be in the range of (0, %s]
 * parameter0,parameter1
 * */
const std::string EM_AICORENUM_OUT_OF_RANGE = "E20103";

/* Failed to open file [%s].
 * parameter0
 * */
const std::string EM_OPEN_FILE_FAILED = "E21001";

/* Failed to read file [%s]. Reason: %s.
 * parameter0,parameter1
 * */
const std::string EM_READ_FILE_FAILED = "E21002";

const std::string EM_FAILED_TO_TOPO_SORTING = "E2100C";

const std::string EM_GRAPH_FUSION_FAILED = "E2100D";

const std::string EM_GRAPH_PASS_OWNER_INVALID = "E2100E";

const std::string EM_TAG_NO_CONST_FOLDING_FAILED = "E2100F";

const std::string EM_COMMON_NULL_PTR = "E21010";

const std::string EM_INVALID_IMPLEMENTATION = "E21011";

const std::string EM_INNER_ERROR_1 = "E21012";

const std::string EM_INVALID_OUTPUT_NAME_INDEX = "E21013";

const std::string EM_INVALID_TENSOR_NAME_INDEX = "E21014";

const std::string EM_FAILED_TO_ASSEMBLE_TBE_INFO = "E21015";

const std::string EM_FAILED_TO_ASSEMBLE_INPUT_INFO = "E21016";

const std::string EM_FAILED_TO_ASSEMBLE_OUTPUT_INFO = "E21017";

const std::string EM_INVALID_TENSOR_DATA_TYPE = "E21018";

const std::string EM_INVALID_ATTR_DATA_TYPE = "E21019";

const std::string EM_SELECT_OP_FORMAT = "E21019";

const std::string EM_FAILED_TO_PARSE_FORMAT_JSON = "E2101A";

const std::string EM_FAILED_TO_CONVERT_FORMAT_FROM_JSON = "E2101B";

const std::string EM_FORMAT_VECTOR_SIZE_INVALID = "E2101C";

const std::string EM_INVALID_FORMAT_IN_JSON = "E2101D";

const std::string EM_INVALID_DTYPE_IN_JSON = "E2101E";

const std::string EM_ORIGINAL_DATATYPE_IS_NOT_SUPPORTED = "E21020";
}

#endif  // FUSION_ENGINE_INC_COMMON_FE_ERROR_CODE_H_
