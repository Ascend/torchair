export WORLD_SIZE=8
export MODEL_NAME="deepseek_v3"
export MODEL_DIR=$1
export INPUT_MAX_LEN=24
export MAX_NEW_TOKENS=64
export BATCH_SIZE=1
export EXE_MODE="dynamo"
export ENABLE_ACLGRAPH=0

cann_path=/usr/local/Ascend/latest
source $cann_path/bin/setenv.bash
export ASCEND_HOME_PATH=$cann_path
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0

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
            --model_name=${MODEL_NAME} --model_path=${MODEL_DIR} --tokenizer_mode=chat \
            --input_max_len=${INPUT_MAX_LEN} --max_new_tokens=${MAX_NEW_TOKENS} --batch_size=${BATCH_SIZE} \
            --execute_mode=${EXE_MODE} &> ${RES_PATH}/log_${LOCAL_RANK}.log &
done
