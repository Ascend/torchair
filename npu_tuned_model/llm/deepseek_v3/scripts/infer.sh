export WORLD_SIZE=8
export MODEL_NAME="deepseek_v3"
export MODEL_DIR=$1
export INPUT_MAX_LEN=1024
export MAX_NEW_TOKENS=32
export BATCH_SIZE=1
export EXE_MODE="eager"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=6038

for((i=0; i<${WORLD_SIZE}; i++))
do
    export LOCAL_RANK=$i
    export RANK_ID=$i
    python3 infer.py \
            --model_name=${MODEL_NAME} --model_path=${MODEL_DIR} \
            --input_max_len=${INPUT_MAX_LEN} --max_new_tokens=${MAX_NEW_TOKENS} --batch_size=${BATCH_SIZE} \
            --execute_mode=${EXE_MODE} &
done