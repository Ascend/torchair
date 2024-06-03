# 介绍

本章节旨在提供在npu上优化改造好的大模型，方便开发者直接使用已经适配好的npu大模型或者对自定义的大模型进行NPU的迁移时提供参考。

是一个快速且易于让大模型接入昇腾CANN框架的推理和服务库

性能优势体现：

- 支持Npu融合算子

易用性优势体现：

- 支持pytorch前端框架

支持的Hugging Face大模型:

- llama2

# 公告

- 2024年5月6号：提供llama2基于分离部署api的适配样例
- 2024年3月6号：提供llama2适配好的npu模型结构和前端切分分布式执行样例

# 环境准备

请参考[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)安装对应的驱动/固件/软件包，在安装深度学习框架时选择安装PyTorch框架，根据资料指导进行安装并设置环境变量，涉及的软件包如下：

|    软件    |                             版本                             |
| :--------: | :----------------------------------------------------------: |
|   Python   |                            3.8.0                             |
|   driver   | [社区版本](https://www.hiascend.com/software/cann/community) |
|  firmware  | [社区版本](https://www.hiascend.com/software/cann/community) |
|    CANN    | [社区版本](https://www.hiascend.com/software/cann/community) |
|   kernel   | [社区版本](https://www.hiascend.com/software/cann/community) |
|   torch    |                            2.1.0                             |
| torch_npu  |                            2.1.0                             |
|    apex    |                             0.1                              |
| 第三方依赖 |                       requirement.txt                        |

# 环境搭建

```shell
# arm环境搭建示例
conda create -n test python=3.8
conda activate test

# 根据CANN安装指南安装固件/驱动/cann包

# 安装 torch 和 torch_npu
pip3 install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
pip3 install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

pip3 install -r requirement.txt

git clone https://gitee.com/ascend/torchair.git
cd torchair/npu_tuned_model/llm
bash setup.sh
```

**注意**：建议昇腾run包安装在conda创建的镜像中安装。如果遇到ASCEND_HOME_PATH环境变量未设置，设置为run包安装路径

# 项目结构

```tex
├── README.md
├── chatglm
│   ├── README.md
│   ├── benchmark_chatglm.py
│   └── modeling_chatglm.py
├── examples
│   ├── __init__.py
│   ├── llama
│   │   ├── run_llama.py
│   │   └── separate_deployment
│   │       ├── run_decoder.py
│   │       ├── run_prompt.py
│   │       └── 分离部署迁移指南.md
│   └── merge_qkv_weight.py
├── models
│   ├── README.md
│   ├── __init__.py
│   ├── common
│   │   ├── mc2_adapter.py
│   │   └── utils.py
│   └── llama
│       ├── README.md
│       └── modeling_llama.py
├── requirement.txt
├── runner
│   ├── __init__.py
│   ├── common_runner.py
│   ├── llm_runner.py
│   └── separate_deployment
│       ├── llm_decoder.py
│       ├── llm_inference.py
│       └── llm_prompt.py
└── setup.sh
```

# 模型及数据集

| 数据集 | 参数量                                                       |
| ------ | ------------------------------------------------------------ |
| llama2 | [70b](https://huggingface.co/TheBloke/Llama-2-70B-fp16/tree/main) |
| llama3 | [70b](https://huggingface.co/meta-llama/Meta-Llama-3-70B/tree/main) |
| chatglm3 | [6b](https://huggingface.co/THUDM/chatglm3-6b) |

# 快速体验
以llama2为例：
```shell
export PYTHONPATH=$PYTHONPATH:/path/to/your/torchair/npu_tuned_model/llm
cann_path=/usr/local/Ascend # 昇腾cann包安装目录
source ${cann_path}/latest/bin/setenv.bash
model_path=xxx/llama2-70b # 下载的权重和模型信息
python3 examples/merge_qkv_weight.py --model_path=${model_path} --model_name=llama2 --tp_size=8 --output_path=xxx/llama-70b_qkv
# 若想使能mc2融合算子优化，使用models/common/mc2_adapter.py自定义的LinearAllreduce替换原生的deepspeed.module_inject.layers.LinearAllreduce
deepspeed --num_gpus=8 examples/llama2/run_llama2.py --model_path=xxx/llama2-70b_qkv
```

# 使能改造好的npu models

一：先将对应模型的权重进行qkv融合

```shell
model_path=xxx/llama2-70b # 下载的权重和模型信息
python3 examples/merge_qkv_weight.py --model_path=${model_path} --model_name=llama2 --tp_size=8 --output_path=xxx/llama-70b_qkv
```

二：根据执行方式不同

- 直接导入LlmModelRunner可以快速执行一个已经改造好的models

```python
from runner.llm_runner import LlmModelRunner
import torch
import os

prompts = [
    "Common sense questions\n\nQuestion:What is a banana?",
    "Common sense questions and answers\n\nQuestion: What is a dog?\nFactual answer:",
    "Common sense questions and answers\nQuestion: How to learn a language?\nFactual answer:",
    "Common sense questions and answers\n\nQuestion: Can you introduce yourself?\nFactual answer:",
]

model_path = "/path/to/your/llama2-70b_qkv"
config = {
    "input_padding": True,
    "dtype": torch.float16,
    "input_max_len": 1024,
    "max_new_tokens": 1024,
}

model_runner = LlmModelRunner("llama2", model_path, **config)  # 可以更换不同的model名字执行不同的大模型

# 设置npu上执行的参数配置
model_runner.execute_mode = "dynamo"
os.environ["EXE_MODE"] = model_runner.execute_mode
model_runner.set_npu_config(jit_compile=False)

# 执行推理
model_runner.execute_model(prompts, **config)
```

- 直接替换用户脚本中的**模型A**为改造后models目录下的**模型A'**。

  **step1**：进行pytorch npu的迁移。根据执行模式选择对应迁移资料进行迁移

  [pytorch npu迁移资料](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/ptonlineinfer/PyTorch_Infer_000001.html)

  [pytorch npu图模式迁移资料](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/torchair/torchair_01_0001.html)(待补充)

  **step2**：找到用户脚本中导入大模型结构的地方，替换成改造后的npu大模型

```python
# 以llama2为例。替换前，导入的是用户脚本中原有的models
from transformers import AutoTokenizer, LlamaForCausalLM
#替换后
from models.llama.modeling_llama import LlamaForCausalLM
```

**注：mc2融合算子优化需要根据模型优化方式手动使能**

# LlmModelRunner加入用户自定义大模型

- 注意当前提供的模型改造和执行样例都是基于deepspeed的前端切分方式
- 根据models下已有的大模型改造样例，参考类似模型的改造点，对自定义的大模型进行模型改造
- 改造完成后，在models目录下新增自定义模型目录，并且创建自定义模型的model文件，同时需要将自定义模型的名字/文件名/class写入\_\_init\_\_.py

```python
# llama2是模型名，llama是模型所在的文件夹，modeling_llama是对于llama目录下modeling_llama.py文件，LlamaForCausalLM是自定义llama2模型
_MODELS = {
    "llama2": ("llama", "modeling_llama", "LlamaForCausalLM"),
    "llama3": ("llama", "modeling_llama", "LlamaForCausalLM"),
}

# 前端切分的大模型还需要指定被切分的layer在哪一层汇聚
# "LlamaDecoderLayer", ('self_attn.o_proj', 'mlp.down_proj')是llama2需要做allreduce的layer
_MODEL_INJECTION_POLICY = {
    "llama2": ("llama", "modeling_llama", "LlamaDecoderLayer", ('self_attn.o_proj', 'mlp.down_proj')),
    "llama3": ("llama", "modeling_llama", "LlamaDecoderLayer", ('self_attn.o_proj', 'mlp.down_proj')),
}
```

- 新增模型执行脚本

  在examples目录下创建自己大模型目录，并新增run_llama2.py的模型调用脚本，示例如下

  [^注意事项]: example提供的执行样例是基于transformers集成的对外接口执行的

```python
import os
import argparse
import logging
import torch
from runner.llm_runner import LlmModelRunner

prompts = [
    "Common sense questions\n\nQuestion:What is a banana?",
    "Common sense questions and answers\n\nQuestion: What is a dog?\nFactual answer:",
    "Common sense questions and answers\nQuestion: How to learn a language?\nFactual answer:",
    "Common sense questions and answers\n\nQuestion: Can you introduce yourself?\nFactual answer:",
]

def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--model_path', type=str, help="Location of model weights")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id")
    parser_args = parser.parse_args()
    return parser_args

def run_llama2(model_path, **kwargs):
    model_runner = LlmModelRunner("llama2", model_path, **kwargs)
    model_runner.execute_mode = "dynamo"
    os.environ["EXE_MODE"] = model_runner.execute_mode
    model_runner.set_npu_config(jit_compile=False)
    model_runner.execute_model(prompts, **kwargs)

if __name__ == "__main__":
    args = parse_args()
    input_padding = True
    config = {
        "input_padding": input_padding,
        "dtype": torch.float16,
        "input_max_len": 1024,
        "max_new_tokens": 1024,
    }
    run_llama2(args.model_path, **config)
    logging.info("model run success")
```

同时可以新创建LlmModelRunner的子类，对其中函数方法中涉及模型部分进行自定义修改

- 模型执行

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/your/torchair/npu_tuned_model/llm
cann_path=/usr/local/Ascend #昇腾cann包安装目录
source ${cann_path}/latest/bin/setenv.bash
deepspeed --num_gpus=8 examples/llama/run_llama.py --model_path=/path/to/your/model/weight
```

# 使能分离部署功能

详情参考[分离部署迁移指南](./examples/llama2/separate_deployment/分离部署迁移指南.md)

