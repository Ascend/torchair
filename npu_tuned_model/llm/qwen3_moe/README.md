# Qwen3-Moe

本sample主要是Qwen3Moe模型在npu上的推理适配点介绍，使用transformers==4.46.3版本，基于Qwen3Moe开源算法[modeling_qwen3_moe.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)进行迁移。

---

# 1. Quick Start：执行样例

本sample的目录下提供了手动Tensor并行及Qwen3Moe推理的执行样例参考

## 1.1. 环境准备
**安装对应的transformers版本与numpy版本**

```shell
pip3 install transformers==4.46.3
pip3 install numpy==1.24.4
```

**安装torch_npu**

相关介绍与安装步骤参考：[torch_npu](https://gitee.com/ascend/pytorch)

> torch_npu是支持PyTorch框架运行在昇腾NPU上的适配插件，提供AI模型执行在昇腾NPU上的基础能力。
> 同时我们使用其torchair子模块提供的torch.compile后端，加速模型执行。相关介绍参考：[torchair](https://gitee.com/ascend/torchair)
> 版本配套关系参考：[附录2：版本配套表](#5-附录2版本配套表)

**安装MindSpeed**

相关介绍与安装步骤参考：[MindSpeed](https://gitee.com/ascend/MindSpeed)

> MindSpeed 是针对华为昇腾设备的大模型加速库。我们主要使用其提供的GMM融合算子，加速Qwen3Moe模型MOE专家计算部分的执行

```shell
git clone https://gitee.com/ascend/MindSpeed.git
pip3 install -e MindSpeed
```

**设置环境变量**

```shell
cann_path=/usr/local/Ascend # 昇腾cann包安装目录
source ${cann_path}/latest/bin/setenv.bash
export ASCEND_HOME_PATH=${cann_path}/latest
export WORLD_SIZE=16 # 多卡推理时使用的卡数，默认为16
```

## 1.2. 权重准备

下载[Qwen3Moe权重](https://huggingface.co/Qwen/Qwen3-235B-A22B/tree/main)到本地路径`model_path_origin`

**手动切分与调整权重**

本sample对Qwen3Moe开源模型进行了部分结构调整，需要对原始权重进行切分与调整。通过执行`scripts/split_weight.py`进行切分与调整，涉及参数如下：
- 已提供权重切分脚本`scripts/split_weight.py`
- `model_path_origin`为原始完整权重路径
- `model_path_tp`为切分与调整后的权重路径
- `world_size`为切分权重的份数，即多卡推理时使用的卡数。默认为16，即默认执行16卡切分与调整

```shell
python3 scripts/split_weight.py --model-path model_path_origin --output-path model_path_tp --world-size world_size
```

## 1.3. Qwen3Moe多卡推理

基于已切分好的权重路径`model_path_tp`，可通过已提供的推理启动脚本`scripts/infer.sh`进行多卡推理:
```shell
cd scripts
bash infer.sh model_path_tp
```
**需要注意**
默认的推理脚本通过调用推理执行脚本`scripts/infer.py`实现：
- 提供16卡推理
- 默认输入padding到1024，最大输出token数为32，decode为单batch推理。
- 默认使用`eager`单算子模式推理
- 相关参数说明参考[附录1：环境变量说明](#4-附录1环境变量说明)

---

# 2. 目录结构

本sample目录结构与文件介绍如下：
- `engine`目录：涉及通用模型执行引擎`model_run.py`
  - `model_run.py`：模型执行引擎，包含模型初始化、模型加载、tokenizer初始化、模型推理等通用基类方法。
- `scripts`目录：涉及当前Qwen3Moe模型执行涉及的相关脚本
  - `models`目录：涉及模型脚本
    - `configuration_qwen3_moe.py`：Qwen3Moe模型配置config
    - `modeling_qwen3_moe.py`：Qwen3Moe模型脚本
  - `runner_qwen3_moe.py`：基于通用模型执行引擎进行继承，适配当前模型所需各项内容
  - `split_weight.py`：权重切分工具
  - `split_weight.sh`：权重切分启动脚本
  - `infer.py`：Qwen3Moe模型推理执行脚本
  - `infer.sh`：Qwen3Moe模型推理启动脚本

---