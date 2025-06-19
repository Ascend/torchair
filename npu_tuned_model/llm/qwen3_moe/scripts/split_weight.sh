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

cann_path=/usr/local/Ascend/ascend-toolkit/latest
source $cann_path/bin/setenv.bash
export ASCEND_HOME_PATH=$cann_path

export WORLD_SIZE=16
export LAYER_NUM=94

path_model_origin=$1
path_model_after_tp=$2

python split_weight.py --model-path ${path_model_origin} --output-path ${path_model_after_tp} --world-size ${WORLD_SIZE}