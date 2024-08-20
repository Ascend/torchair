# 介绍

本章节旨在提供在npu上优化改造好的大模型，方便开发者直接使用已经适配好的npu大模型或者对自定义的大模型进行NPU的迁移时提供参考。

是一个快速且易于让大模型接入昇腾CANN框架的推理和服务库

性能优势体现：

- 支持Npu融合算子

易用性优势体现：

- 支持pytorch前端框架

支持的Hugging Face大模型:

- qwen-vl

# 公告

- 2024年8月20号：新增qwen-vl模型中vit模块优化项，新增优化后的性能数据
- 2024年7月28号：提供qwen-vl适配好的npu模型结构和执行样例

# 环境准备

请参考[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)安装对应的驱动/固件/软件包，在安装深度学习框架时选择安装PyTorch框架，根据资料指导进行安装并设置环境变量，涉及的软件包如下：

|    软件    |                            版本                            |
| :--------: |:--------------------------------------------------------:|
|   Python   |                          3.8.0                           |
|   driver   | [社区版本](https://www.hiascend.com/software/cann/community) |
|  firmware  | [社区版本](https://www.hiascend.com/software/cann/community) |
|    CANN    | [社区版本](https://www.hiascend.com/software/cann/community) |
|   kernel   | [社区版本](https://www.hiascend.com/software/cann/community) |
|   torch    |                          2.1.0                           |
| torch_npu  |                          2.1.0                           |
|    apex    |                           0.1                            |
| 第三方依赖 |                  [requirement.txt](./requirement.txt)                   |

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

git clone https://gitee.com/ascend/torchair.git
cd torchair/npu_tuned_model/mm/qwen-vl
pip3 install -r requirement.txt
```

**注意**：建议昇腾run包安装在conda创建的镜像中安装。如果遇到ASCEND_HOME_PATH环境变量未设置，设置为run包安装路径

# 下载模型

| 模型      | 版本                                           |
|---------|----------------------------------------------|
| qwen-vl | [base](https://huggingface.co/Qwen/Qwen-VL) |

# 快速体验

[qwen-vl](./model/README.md#性能测试)

# 新模型改造

选择和样例模型结构近似的进行参考适配
