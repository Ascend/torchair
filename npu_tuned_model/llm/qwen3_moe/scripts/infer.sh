# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export WORLD_SIZE=16
export MODEL_NAME="qwen3_moe"
export MODEL_DIR=$1
export INPUT_MAX_LEN=4080
export MAX_NEW_TOKENS=16
export BATCH_SIZE=1
export EXE_MODE="eager"
export ENABLE_ACLGRAPH=0
export ENABLE_PROFILE=0
export LAYER_NUM=94

cann_path=/usr/local/Ascend/ascend-toolkit/latest
source $cann_path/bin/setenv.bash
export ASCEND_HOME_PATH=$cann_path
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export HCCL_OP_EXPANSION_MODE=AIV

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=6038

DATE=`date +%Y%m%d`
DIR_PREFIX="res"
NAME=${MODEL_NAME}_${WORLD_SIZE}_${EXE_MODE}_B${BATCH_SIZE}_In${INPUT_MAX_LEN}_Out${MAX_NEW_TOKENS}
export RES_PATH="${DIR_PREFIX}/${DATE}/${NAME}"
mkdir -p ${RES_PATH}
echo "res_path=" $RES_PATH

for((i=0; i<${WORLD_SIZE}; i++))
do
    export LOCAL_RANK=$i
    export RANK_ID=$i
    python3 infer.py \
            --model_name=${MODEL_NAME} --model_path=${MODEL_DIR} --tokenizer_mode=default \
            --input_max_len=${INPUT_MAX_LEN} --max_new_tokens=${MAX_NEW_TOKENS} --batch_size=${BATCH_SIZE} \
            --execute_mode=${EXE_MODE} &> ${RES_PATH}/log_${LOCAL_RANK}.log &
done
