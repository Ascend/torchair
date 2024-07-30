export PYTHONPATH=$PYTHONPATH:/your/path/to/torchair/npu_tuned_model/mm
cann_path=/usr/local/Ascend # 昇腾cann包安装目录
model_path=xxx/stable-diffusion-2-1-base # 下载的权重和模型信息
device_list="0"
args=$@

for arg in ${args}
do
    if [[ "${arg}" =~ "--cann_path=" ]];then
        cann_path=${arg#*=}
    elif [[ "${arg}" =~ "--model_path=" ]];then
        model_path=${arg#*=}
    elif [[ "${arg}" =~ "--device_list=" ]];then
        device_list=${arg#*=}
    fi
done

device_array=(${device_list//,/ })
device_num=${#device_array[@]}
echo ${device_num}

export WORLD_SIZE=${device_num}
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="8008" #配置未被占用的端口
index=0
# source ${cann_path}/latest/bin/setenv.bash

for device in ${device_array[*]};
do
    export RANK_ID=${index}
    export LOCAL_RANK=${index}
    let index=index+1
    echo ${RANK_ID}
    if [ ${index} -eq ${device_num} ];then
        python3.8 run_sd.py --model_path=${model_path} --local_rank=${RANK_ID} --FA --dynamo --TOME --DC
    else
        python3.8 run_sd.py --model_path=${model_path} --local_rank=${RANK_ID} --FA --dynamo --TOME --DC &
    fi
done